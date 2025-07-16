import torch
from PIL import Image
from pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from accelerate.utils import set_seed
import pdb

# ----------------------------------------------------------------------------------------------------------------------------
# import debugpy
# try:
#     debugpy.listen(("0.0.0.0", 5678))     # debugpy.listen(("0.0.0.0", 5678))   # debugpy.listen(("127.0.0.1", 5678))
# except RuntimeError as e:
#     print(f"debugpy already listening: {e}")
# print("🧩 Waiting for debugger to attach...")
# debugpy.wait_for_client()

# image = pipe(
#     "A cat holding a sign that says hello world",
#     negative_prompt="",
#     num_inference_steps=28,
#     guidance_scale=7.0,
# ).images[0]
# ----------------------------------------------------------------------------------------------------------------------------
seed = 0
inference_steps = [14, 28, 56]
inversion_guidance_scale = [1.0, 1.0, 1.0, 1.0, 1.0]
denoising_guidance_scale = [1.0, 1.5, 2.0, 2.5, 3.0]
image_path = "/home/yhyun225/BiRF/examples/horse.jpg"
prompt = "A young boy is riding a brown horse in a countryside field, with a large tree in the background."
# image_path = "/home/yhyun225/BiRF/examples/boy.jpg"
# prompt = "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above."
# image_path = "/home/yhyun225/BiRF/examples/cat.png"
# prompt = "A cat holding a sign that says hello world"
use_lora = True

# 0) set seed for deterministic experimental result
set_seed(seed)

# 1) Load Stable Diffusion 3 pipeline
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder='scheduler', cache_dir='/hdd/yhyun225/cache')
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, scheduler=scheduler, cache_dir='/hdd/yhyun225/cache'
)
pipe = pipe.to("cuda")

# 2) Load lora weights into the pipeline
# if use_lora:
#     pipe.load_lora_weights('/home/yhyun225/BiRF/output/ckpts/pytorch_lora_weights.safetensors')

# 3) Reconstruction with inversion & denoising
original_sample = Image.open(image_path)

for step in inference_steps:
    for igs, dgs in zip(inversion_guidance_scale, denoising_guidance_scale):
        # 1st order #
        if use_lora:
            pipe.load_lora_weights('/home/yhyun225/BiRF/output/ckpts/pytorch_lora_weights.safetensors')
        
        latents = pipe.inversion(
            prompt,
            negative_prompt="",
            num_inference_steps=step,
            guidance_scale=igs,
            images=original_sample,
            output_type='latent',
        ).images[0]

        # with torch.no_grad():
        #     _latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        #     recons_image = pipe.vae.decode(_latents.unsqueeze(0), return_dict=False)[0]
        #     recons_image = pipe.image_processor.postprocess(recons_image, output_type="pil")[0]
            
        #     recons_image.save(f"latents_lora_{use_lora}_1st_order_{step}steps_gs_{igs}_{dgs}.png")

        if use_lora:
            pipe.unload_lora_weights()

        image = pipe(
            prompt,
            latents=latents.unsqueeze(0),
            negative_prompt="",
            num_inference_steps=step,
            guidance_scale=dgs,
            output_type="pil",
        ).images[0]

        image.save(f"lora_{use_lora}_1st_order_{step}steps_gs_{igs}_{dgs}.png")

        # 2nd order #
        if use_lora:
            pipe.load_lora_weights('/home/yhyun225/BiRF/output/ckpts/pytorch_lora_weights.safetensors')

        latents = pipe.rf_solver(
            prompt,
            negative_prompt="",
            num_inference_steps=step,
            guidance_scale=igs,
            images=original_sample,
            output_type='latent',
            inversion=True,
            log_inversion=False,
        ).images[0]

        # with torch.no_grad():
        #     _latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

        #     recons_image = pipe.vae.decode(_latents.unsqueeze(0), return_dict=False)[0]
        #     recons_image = pipe.image_processor.postprocess(recons_image, output_type="pil")[0]
        #     recons_image.save(f"latents_lora_{use_lora}_2nd_order_{step}steps_gs_{igs}_{dgs}.png")

        if use_lora:
            pipe.unload_lora_weights()

        image = pipe.rf_solver(
            prompt,
            latents=latents.unsqueeze(0),
            negative_prompt="",
            num_inference_steps=step,
            guidance_scale=dgs,
            output_type="pil",
            inversion=False,
        ).images[0]

        image.save(f"lora_{use_lora}_2nd_order_{step}steps_gs_{igs}_{dgs}.png")
