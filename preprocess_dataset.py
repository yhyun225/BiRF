import os
import torch
import numpy as np
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from dataset import YePopDataset
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, T5TokenizerFast
from transformers import CLIPTextModelWithProjection, T5EncoderModel

from utils.encoding_utils import encode_prompt, encode_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='configs/preprocess_yepop.yaml',
    )
    parser.add_argument(
        '--output_dir', type=str, default='output',
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
    )
    parser.add_argument(
        '--batch_size_per_gpu', type=int, default=None,
    )
    parser.add_argument(
        '--mixed_precision', type=str, choices=['fp16', 'fp32'], default='fp32',
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    accelerator_project_config = ProjectConfiguration()
    accelerator = Accelerator(
        mixed_precision='fp16' if args.mixed_precision =='fp16' else 'no',
        project_config=accelerator_project_config,
    )

    batch_size = args.batch_size
    batch_size_per_gpu = args.batch_size_per_gpu
    assert batch_size is not None or batch_size_per_gpu is not None, \
        'either batch_size or batch_size_per_gpu should be specified'
    batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None \
                                                    else batch_size // accelerator.num_processes
    
    dataset = YePopDataset(cfg.dataset, return_image_path=True)
    if accelerator.is_main_process:
        print(dataset)

    # create 'latent' & '{caption_model}_text_emb' subdirectories under the root path
    root_path = dataset.root_path
    caption_model = dataset.caption_model_type

    if accelerator.is_main_process:
        print('Creating subdirectories...', end='')
        chunks = sorted(os.listdir(root_path))
        for i, chunk in enumerate(chunks):
            chunk_latent_path = os.path.join(*[root_path, chunk, 'latents'])
            chunk_text_emb_path = os.path.join(*[root_path, chunk, f'{caption_model}_text_emb'])
            os.makedirs(chunk_latent_path, exist_ok=True)
            os.makedirs(chunk_text_emb_path, exist_ok=True)
            os.makedirs(os.path.join(chunk_text_emb_path, 'prompt_embed'), exist_ok=True)
            os.makedirs(os.path.join(chunk_text_emb_path, 'pooled_prompt_embed'), exist_ok=True)
        print(' done!')
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=cfg.dataloader.drop_last,
        persistent_workers=cfg.dataloader.persistent_workers,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
        shuffle=cfg.dataloader.shuffle,
    )

    dataloader = accelerator.prepare(dataloader)

    weight_dtype = torch.float16 if args.mixed_precision == 'fp16' else torch.float32
    pretrained_model_name_or_path = cfg.model.pretrained_model_name_or_path

    # prepare VAE for preprocessing images into latents
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder='vae', torch_dtype=weight_dtype, cache_dir=cfg.model.cache_dir,
    ).to(accelerator.device)

    vae.requires_grad_(False)

    # prepare tokenizer & text_encoders for preprocessing texts into text_embeddings
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder='tokenizer', cache_dir=cfg.model.cache_dir,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder='tokenizer_2', cache_dir=cfg.model.cache_dir,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, subfolder='tokenizer_3', cache_dir=cfg.model.cache_dir,
    )
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]

    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder='text_encoder', torch_dtype=weight_dtype, cache_dir=cfg.model.cache_dir,
    ).to(accelerator.device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder='text_encoder_2', torch_dtype=weight_dtype, cache_dir=cfg.model.cache_dir,
    ).to(accelerator.device)
    text_encoder_three = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path, subfolder='text_encoder_3', torch_dtype=weight_dtype, cache_dir=cfg.model.cache_dir,
    ).to(accelerator.device)

    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    if accelerator.is_main_process:
            print('***** Preprocessing dataset *****')
            print(f"  - Num examples: {len(dataset)}")
            print(f"  - Num batches each epoch: {len(dataloader)}")
            print(f"  - Num Epochs: 1")
            print(f"  - Effective batch size: {batch_size_per_gpu * accelerator.num_processes}")
            print(f"    - Batch size per device: {batch_size_per_gpu}")

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        pixel_values = batch['pixel_values'].to(vae.dtype)
        text_prompts = batch['text_prompts']
        image_paths = batch['image_path']

        
        with torch.no_grad():
            latents = encode_image(vae, pixel_values, normalized=True)
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders=text_encoders,
                tokenizers=tokenizers,
                prompt=text_prompts
            )

        for idx, image_path in enumerate(image_paths):
            latent = latents[idx]
            prompt_embed = prompt_embeds[idx]
            pooled_prompt_embed = pooled_prompt_embeds[idx]
            
            image_id = image_path.split('/')[-1].split('.')[0]

            latent_path = os.path.join(
                *[Path(image_path).parent.parent, 'latents', image_id + '.pt']
            )

            prompt_embed_path = os.path.join(
                *[Path(image_path).parent.parent, f'{caption_model}_text_emb', 'prompt_embed', image_id + '.pt']
            )

            pooled_prompt_embed_path = os.path.join(
                *[Path(image_path).parent.parent, f'{caption_model}_text_emb', 'pooled_prompt_embed', image_id + '.pt']
            )

            torch.save(latent.detach().cpu(), latent_path)
            torch.save(prompt_embed.detach().cpu(), prompt_embed_path)
            torch.save(pooled_prompt_embed.detach().cpu(), pooled_prompt_embed_path)



if __name__ == "__main__":
     main()