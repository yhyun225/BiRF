import os
import sys
import argparse
import yaml
from omegaconf import OmegaConf

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from diffusers.utils import is_wandb_available

from trainer import Trainer
from dataset import YePopDataset

from huggingface_hub import whoami, login
from utils.logger_utils import setup_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='configs/train_sd3_lora.yaml',
    )
    parser.add_argument(
        '--output_dir', type=str, default='output',
    )
    parser.add_argument(
        '--seed', type=int, default=0,
    )
    parser.add_argument(
        '--max_steps', type=int, default=100_000,
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
    )
    parser.add_argument(
        '--batch_size_per_gpu', type=int, default=None,
    )
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=1,
    )
    parser.add_argument(
        '--mixed_precision', type=str, choices=['fp16', 'fp32'], default='fp16'
    )
    parser.add_argument(
        '--i_print', type=int, default=1_000,
    )
    parser.add_argument(
        '--i_log', type=int, default=500,
    )
    parser.add_argument(
        '--i_sample', type=int, default=10_000,
    )
    parser.add_argument(
        '--i_save', type=int, default=10_000
    )
    parser.add_argument(
        '--wandb', action='store_true', default=False
    )
    

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    
    output_dir = args.output_dir
    logging_dir = os.path.join(output_dir, 'log')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    
    logger = setup_logger(logging_dir)
    set_seed(args.seed)
    
    accelerator_project_config = ProjectConfiguration(
            project_dir=output_dir, 
            logging_dir=logging_dir,
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='fp16' if args.mixed_precision == 'fp16' else 'fp32',
        project_config=accelerator_project_config,
        log_with="wandb" if is_wandb_available and args.wandb else None,
    )

    if accelerator.is_main_process:
        try:
            user_info = whoami()
            logger.info(f"***** Huggingface-hub logged in as [{user_info['name']}].")
        except:
            logger.info("***** Not logged in or invalid token")
            logger.info("***** Try login...")
            login()
            
        

        with open(os.path.join(logging_dir, 'command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp)
        with open(os.path.join(logging_dir, 'config.yaml'), 'w') as fp:
            yaml.dump(OmegaConf.to_container(cfg, resolve=True), fp, indent=4)
        
    
    dataset = YePopDataset(cfg.dataset, preprocessed_annotation='preprocessed_annotation.pt')
    # dataset = YePopDataset(cfg.dataset)

    trainer = Trainer(
        accelerator,
        cfg.trainer,
        output_dir=output_dir,
        logging_dir=logging_dir,
        dataset=dataset,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        batch_size_per_gpu=args.batch_size_per_gpu,
        mixed_precision=args.mixed_precision,
        i_print=args.i_print,
        i_log=args.i_log,
        i_sample=args.i_sample,
        i_save=args.i_save,
    )

    trainer.run()

if __name__ == "__main__":
    main()