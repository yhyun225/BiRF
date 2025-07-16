import os
import math

from PIL import Image
import torch
import torchvision.transforms as T

def PSNR(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return torch.mean(20 * torch.log10(1.0 / torch.sqrt(mse))).item()

to_tensor = T.ToTensor()
steps = [14, 28, 56]
inversion_guidance_scale = [1.0, 1.0, 1.0, 1.0]
denoising_guidance_scale = [1.0, 1.5, 2.0, 2.5]
image2_root_path = "/home/yhyun225/BiRF/pipeline/output/example3"

image1_path = "/home/yhyun225/BiRF/examples/boy.jpg"
image1 = to_tensor(Image.open(image1_path))


print("*** LoRA x - 1st order ***")
for step in steps:
    for igs, dgs in zip(inversion_guidance_scale, denoising_guidance_scale):
        image2_path = f"{image2_root_path}/no_lora/lora_False_1st_order_{step}steps_gs_{igs}_{dgs}.png"

        try:
            image2 = to_tensor(Image.open(image2_path))
            print(f"{step}_steps_{igs}_{dgs}: {PSNR(image1, image2)}")
        except:
            continue

print("*** LoRA x - 2nd order ***")
for step in steps:
    for igs, dgs in zip(inversion_guidance_scale, denoising_guidance_scale):
        image2_path = f"{image2_root_path}/no_lora/lora_False_2nd_order_{step}steps_gs_{igs}_{dgs}.png"

        try:
            image2 = to_tensor(Image.open(image2_path))
            print(f"{step}_steps_{igs}_{dgs}: {PSNR(image1, image2)}")
        except:
            continue

print("*** LoRA o - 1st order ***")
for step in steps:
    for igs, dgs in zip(inversion_guidance_scale, denoising_guidance_scale):
        image2_path = f"{image2_root_path}/lora/lora_True_1st_order_{step}steps_gs_{igs}_{dgs}.png"

        try:
            image2 = to_tensor(Image.open(image2_path))
            print(f"{step}_steps_{igs}_{dgs}: {PSNR(image1, image2)}")
        except:
            continue

print("*** LoRA o - 2nd order ***")
for step in steps:    
    for igs, dgs in zip(inversion_guidance_scale, denoising_guidance_scale):
        image2_path = f"{image2_root_path}/lora/lora_True_2nd_order_{step}steps_gs_{igs}_{dgs}.png"

        try:
            image2 = to_tensor(Image.open(image2_path))
            print(f"{step}_steps_{igs}_{dgs}: {PSNR(image1, image2)}")
        except:
            continue
