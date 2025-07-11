import os
import logging
import math
import argparse

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import SD3Transformer2DModel
from pipeline import StableDiffusion3Pipeline


from peft import LoraConfig
from huggingface_hub import whoami, login


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='configs/inference_sd3_lora.yaml'
    )
    parser.add_argument(
        '--output_dir', type=str, default='output/samples',
    )
    parser.add_argument(
        '--seed', type=int, default=23,
    )
    parser.add_argument(
        '--mixed_precision', type=str, choices=['fp16', 'fp32'], default='fp16'
    )
    parser.add_argument(
        '--ckpt_path', type=str, default='output/ckpts/checkpoint-5000'
    )
    parser.add_argument(
        '--data_path', type=str, default='examples'
    )
    parser.add_argument(
        '--inversion', action='store_true', default=False,
        help='Whether to '
    )
    parser.add_argument(
        '--sample', action='store_true', default=False,
    )
    
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    set_seed(args.seed)

    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir,
    )
    accelerator = Accelerator(
        project_config=accelerator_project_config,
        mixed_precision=args.mixed_precision,
    )
    
    try:
        user_info = whoami()
        print(f"***** Huggingface-hub logged in as [{user_info['name']}].")
    except:
        print("***** Not logged in or invalid token.")
        print("***** Try login...")
        login()

    # load accelerate state
    # NOTE: lora is stored within transformer here,
    # we should either load the transformer to the pipeline or,
    # detach lora from transformer and then set lora to the pipeline.
    