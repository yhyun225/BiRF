import os
import time
import copy
import logging
import math
import itertools

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

import diffusers
from diffusers.optimization import get_scheduler
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline
)
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)

from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from transformers import CLIPTokenizer, T5TokenizerFast
from transformers import CLIPTextModelWithProjection, T5EncoderModel

from utils.encoding_utils import tokenize_prompt, encode_prompt 
from utils.logger_utils import calculate_eta

if is_wandb_available():
    import wandb

logger = logging.getLogger('mylogger')

class Trainer(object):
    def __init__(
        self,
        accelerator,
        cfg,
        output_dir,
        logging_dir,
        dataset,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        prefetch_data=True,
        mixed_precision='fp16',
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        **kwargs
    ):
        self.accelerator = accelerator
        self.is_master = self.accelerator.is_main_process
        self.world_size = self.accelerator.num_processes

        self.cfg = cfg

        self.dataset = dataset
        self.max_steps = max_steps

        assert batch_size is not None or batch_size_per_gpu is not None, \
            'either batch_size or batch_size_per_gpu should be specified'
        self.batch_size = batch_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None \
                                                    else batch_size // self.world_size
        
        self.prefetch_data = prefetch_data
        self.mixed_precision = mixed_precision
        self.weight_dtype = torch.float16 if self.mixed_precision else torch.float32
        self.device = self.accelerator.device

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save

        self.step = None

        if self.is_master:
            self.ckpt_dir = os.path.join(self.output_dir, 'ckpts')
            self.sample_dir = os.path.join(self.output_dir, 'samples')
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.sample_dir, exist_ok=True)

        self.init_models(self.cfg.model)
        self.prepare_dataloader(self.cfg.dataloader)
        self.prepare_optimizer(self.cfg.optimizer)

        self.transformer, self.optimizer, self.dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.transformer, self.optimizer, self.dataloader, self.lr_scheduler
        )

        if self.is_master and is_wandb_available():
            self.accelerator.init_trackers(
                project_name="BiRF", 
                config=OmegaConf.to_container(self.cfg, resolve=True)
            )

        if self.is_master:
            logger.info('\n\nTrainer initialized.')
            logger.info(self)

    def __str__(self):
        lines = []
        lines.append(f'***** {self.__class__.__name__} *****')
        lines.append(f'  - Dataset: {str(self.dataset)}')
        lines.append(f'  - Dataloader:')
        lines.append(f'    - Sample: {self.dataloader.sampler.__class__.__name__}')
        lines.append(f'    - Num workers: {self.dataloader.num_workers}')
        lines.append(f'  - Number of steps: {self.max_steps}')
        lines.append(f'  - Number of GPUs: {self.world_size}')
        lines.append(f'  - Batch size: {self.batch_size}')
        lines.append(f'  - Batch size per GPU: {self.batch_size_per_gpu}')
        lines.append(f'  - Optimizer: {self.optimizer.__class__.__name__}')
        lines.append(f'  - Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        lines.append(f'  - LR scheduler: {self.lr_scheduler.__class__.__name__}')
        lines.append(f'  - FP16 mode: {self.mixed_precision == "fp16"}')
        return '\n'.join(lines)
    
    def init_models(self, model_cfg):
        """
        Initilize models.
        NOTE: preconditioned refers to precomputed latents & text embeddings.
              If not precomputed, we should compute them inside the training loop.
        """
        pretrained_model_name_or_path = model_cfg.pretrained_model_name_or_path
        
        self.frozen_transformer = SD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder='transformer',
        )
        self.transformer = copy.deepcopy(self.frozen_transformer)
        
        self.frozen_transformer.requires_grad_(False)
        self.transformer.requires_grad_(False)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder='scheduler',
        )

        # [TODO] check whether the lora weight is activated.
        if model_cfg.lora.layers is not None:
            target_modules = [layer.strip() for layer in model_cfg.lora_layers.split(",")]
        else:
            target_modules = [
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "attn.to_k",
                "attn.to_out.0",
                "attn.to_q",
                "attn.to_v",
            ]

        transformer_lora_config = LoraConfig(
            r = model_cfg.lora.rank,
            lora_alpha = model_cfg.lora.rank,
            lora_dropout = model_cfg.lora.dropout,
            init_lora_weights ="gaussian",
            target_modules=target_modules,
        )
        self.transformer.add_adapter(transformer_lora_config)
        
        self.frozen_transformer.to(self.device, self.weight_dtype)
        self.transformer.to(self.device, self.weight_dtype)

        # Make sure the trainable params are in float32
        if self.weight_dtype == torch.float16:
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(self.transformer, dtype=torch.float32)

        # VAE
        self.vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path, subfolder='vae',
            )
        self.vae.requires_grad_(False)
        self.vae.to(self.device, self.weight_dtype)

        if not self.dataset.is_preprocessed:
            # Tokenizer
            tokenizer_one = CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path, subfolder='tokenizer', cache_dir=model_cfg.cache_dir,
            )
            tokenizer_two = CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path, subfolder='tokenizer_2', cache_dir=model_cfg.cache_dir,
            )
            tokenizer_three = T5TokenizerFast.from_pretrained(
                pretrained_model_name_or_path, subfolder='tokenizer_3', cache_dir=model_cfg.cache_dir,
            )
            self.tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]


            # Text encoder
            text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
                pretrained_model_name_or_path, subfolder='text_encoder', torch_dtype=self.weight_dtype, cache_dir=model_cfg.cache_dir,
            ).to(self.device)
            text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                pretrained_model_name_or_path, subfolder='text_encoder_2', torch_dtype=self.weight_dtype, cache_dir=model_cfg.cache_dir,
            ).to(self.device)
            text_encoder_three = T5EncoderModel.from_pretrained(
                pretrained_model_name_or_path, subfolder='text_encoder_3', torch_dtype=self.weight_dtype, cache_dir=model_cfg.cache_dir,
            ).to(self.device)

            text_encoder_one.requires_grad_(False)
            text_encoder_two.requires_grad_(False)
            text_encoder_three.requires_grad_(False)
            self.text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]


    def prepare_dataloader(self, dataloader_cfg):
        """
        Prepare dataloader.
        """
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=dataloader_cfg.pin_memory,
            drop_last=dataloader_cfg.drop_last,
            persistent_workers=dataloader_cfg.persistent_workers,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            shuffle=dataloader_cfg.shuffle,
        )
    
    def prepare_optimizer(self, optimizer_config):
        """
        Prepare optimizer.
        """
        transformer_loar_parameters = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
        learning_rate = optimizer_config.learning_rate
        params_to_optimize = [{"params": transformer_loar_parameters, "lr": learning_rate}]

        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )

        self.lr_scheduler = get_scheduler(
            optimizer_config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=optimizer_config.lr_warmup_steps * self.world_size,
            num_training_steps=self.max_steps * self.world_size,
            num_cycles=optimizer_config.lr_num_cycles,
            power=optimizer_config.lr_power,
        )

    def snapshot_dataset(self, num_samples=100):
        """
        Sample images from the dataset
        """
        pass

    @torch.no_grad()
    def snapshot(self, num_samples=64, batch_size=4, verbose=False):
        """
        Sample images from the model
        NOTE: This function should be called by all processes
        """
        if self.is_master:
            logger.info(f'\nSampling {num_samples} images...', end='')

        # Load sd3 pipeline & text descriptions and...
        #   1) w / o finetuning, sampling + inversion
        #   2) w/ finetuning, sampling + inversion

        if self.is_master:
            logger.info(' Done.')

    def load(self):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        pass

    def save(self):
        """
        Save a checkpoint.
        Should be called by main process.
        """
        if self.is_master:
            transformer = self.accelerator.unwrap_model(copy.deepcopy(self.transformer)).to('cpu')
            transformer = transformer._orig_model if is_compiled_module(transformer) else transformer
            transformer = transformer.to(self.weight_dtype)
            transformer_lora_layers = get_peft_model_state_dict(transformer)
        
            StableDiffusion3Pipeline.save_lora_weights(
                save_directory=self.ckpt_dir,
                transformer_lora_layers=transformer_lora_layers,
            )
            logger.info("Saved checkpoint.")
        
        self.accelerator.wait_for_everyone()

    def run(self):
        """
        Train!
        """

        def get_sigmas(timesteps, scheduler, n_dim=4, dtype=torch.float32, device='cpu'):
            sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
            schedule_timesteps = scheduler.timesteps.to(device)
            timesteps = timesteps.to(device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma
    
        num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.accelerator.gradient_accumulation_steps)
        num_epochs = math.ceil(self.max_steps / num_update_steps_per_epoch)

        total_batch_size = self.batch_size_per_gpu \
                           * self.world_size \
                           * self.accelerator.gradient_accumulation_steps
        
        lines = []
        lines.append('***** Running training *****')
        lines.append(f"  - Num examples: {len(self.dataset)}")
        lines.append(f"  - Num batches each epoch: {len(self.dataloader)}")
        lines.append(f"  - Num Epochs: {num_epochs}")
        lines.append(f"  - Effective batch size: {total_batch_size}")
        lines.append(f"    - Batch size per device: {self.batch_size_per_gpu}")
        lines.append(f"    - Gradient accumulation setps: {self.accelerator.gradient_accumulation_steps}")
        lines.append(f"  - Total optimization steps: {self.max_steps}")
        lines = '\n'.join(lines)
        
        if self.is_master:
            logger.info(lines)

        self.global_step = 0
        first_epoch = 0

        # progress_bar = tqdm(
        #     range(0, self.max_steps),
        #     initial=self.global_step,
        #     desc="Steps",
        #     disable=not self.accelerator.is_local_main_process
        # )

        for epoch in range(first_epoch, num_epochs):
            self.transformer.train()
            
            for step, batch in enumerate(self.dataloader):
                iter_start_time = time.time()

                models_to_accumulate = [self.transformer]

                with self.accelerator.accumulate(models_to_accumulate):
                    pixel_values = batch['pixel_values'].to(dtype=self.weight_dtype)
                    text_prompts = batch['text_prompts']
                    # TODO: How about emptry string '' for CFG?

                    if not self.dataset.is_preprocessed:
                        pixel_values = self.vae.encode(pixel_values).latent_dist.sample()
                        prompt_embeds, pooled_prompt_embeds = encode_prompt(
                            text_encoders=self.text_encoders,
                            tokenizers=self.tokenizers,
                            prompt=text_prompts,
                        )
                    else:
                        prompt_embeds, pooled_prompt_embeds = text_prompts['prompt_embeds'], text_prompts['pooled_prompt_embeds']
                    
                    model_input = (pixel_values - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                    model_input = model_input.to(dtype=self.weight_dtype)

                    # timestep: t
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]

                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=self.cfg.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=self.cfg.logit_mean,
                        logit_std=self.cfg.logit_std,
                        mode_scale=self.cfg.mode_scale
                    )
                    
                    indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
                    timesteps = self.noise_scheduler.timesteps[indices].to(device=self.device)

                    sigmas = get_sigmas(
                        timesteps, 
                        self.noise_scheduler, 
                        n_dim=model_input.ndim, 
                        dtype=model_input.dtype, 
                        device=model_input.device
                    )
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                    
                    with torch.no_grad():
                        model_pred = self.frozen_transformer(
                            hidden_states=noisy_model_input,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]
                    
                    # timestep: t + 1
                    noise_next = torch.randn_like(model_input)

                    indices_next = indices + 1
                    timesteps_next = self.noise_scheduler.timesteps[indices_next].to(device=self.device)
                    
                    sigmas = get_sigmas(
                        timesteps_next,
                        self.noise_scheduler,
                        n_dim=model_input.ndim, 
                        dtype=model_input.dtype, 
                        device=model_input.device
                    )
                    noisy_model_input_next = (1.0 - sigmas) * model_input + sigmas * noise_next

                    model_pred_next = self.transformer(
                        hidden_states=noisy_model_input_next,
                        timestep=timesteps_next,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                    
                    # NOTE: weighting here or not?
                    # weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.cfg.weighting_scheme, sigmas=sigmas)
                    # loss = torch.mean(
                    #     (weighting.float() * (model_pred.float() - model_pred_next.float()) ** 2).reshape(model_pred.shape[0], -1),
                    #     1,
                    # )
                    loss = torch.mean(
                        ((model_pred.float() - model_pred_next.float()) ** 2).reshape(model_pred.shape[0], -1),
                        1,
                    )
                    loss = loss.mean()

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        params_to_clip = [p for p in self.transformer.parameters() if p.requires_grad]
                        self.accelerator.clip_grad_norm_(params_to_clip, 1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                iter_end_time = time.time()
                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    # progress_bar.update(1)
                    self.global_step += 1

                    logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}

                    if self.is_master or self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                        if self.global_step % self.i_print == 0:
                            
                            log_msg = f"[{self.global_step}/{self.max_steps}] " \
                                    f"loss: {logs['loss']:.4f}, " \
                                    f"lr: {logs['lr']:.6f}, " \
                                    f"Peak Memory: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} G, " \
                                    f"ETA: {calculate_eta(iter_start_time, iter_end_time, self.global_step, self.max_steps)}"
                            
                            logger.info(log_msg)

                        if self.global_step % self.i_log == 0:
                            self.accelerator.log(logs, step=self.global_step)

                        if self.global_step % self.i_sample == 0:
                            pass

                        if self.global_step % self.i_save == 0:
                            save_path = os.path.join(self.ckpt_dir, f'checkpoint-{self.global_step}')
                            self.accelerator.save_state(save_path)
                            logger.info('Saved training states.')
                            # self.save()

                if self.global_step >= self.max_steps:
                    break
                a = int(1.2)
        # save lora layers
        self.save()
        
        self.accelerator.end_training()
        logger.info("done!")