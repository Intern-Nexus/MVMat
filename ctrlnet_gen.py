import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from ip_adapter import IPAdapter


def print_green(text):
    print(f"\033[92m{text}\033[0m")

def generate_image_prompt_via_normal_controlnet(normal_path, prompt, controlnet_cond_mode, seed):
    normal = Image.open(normal_path)
        
    if controlnet_cond_mode == 'text':
        sd_ckpt = "runwayml/stable-diffusion-v1-5"
        ctrlnet_ckpt = "lllyasviel/control_v11p_sd15_normalbae"

        control_image = normal

        control_mask = np.where(np.array(control_image) > 0, 1.0, 0.0)

        controlnet = ControlNetModel.from_pretrained(ctrlnet_ckpt, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=sd_ckpt, 
            controlnet=controlnet, torch_dtype=torch.float16
        )
        print_green("[INFO] Text-guided normal ControlNet is loaded.")

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        generator = torch.manual_seed(seed=seed)
        image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

        image = Image.fromarray((np.array(image, dtype=np.float32) * control_mask).astype(np.uint8))
        image.save(f"{os.path.dirname(normal_path)}/ip.png")
    
    elif controlnet_cond_mode == 'image':
        sd_ckpt = "runwayml/stable-diffusion-v1-5"
        ctrlnet_ckpt = "lllyasviel/control_v11p_sd15_normalbae"
        vae_ckpt = "stabilityai/sd-vae-ft-mse"
        image_encoder_ckpt = "ip_adapter/models/image_encoder/"
        ip_ckpt = "ip_adapter/models/ip-adapter_sd15.safetensors"
        device = "cuda"

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(vae_ckpt).to(dtype=torch.float16)

        # load controlnet
        controlnet = ControlNetModel.from_pretrained(ctrlnet_ckpt, torch_dtype=torch.float16)

        # load SD pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_ckpt,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )

        # read image prompt
        prompt = Image.open(prompt)
        mask = np.where(np.array(normal, dtype=np.float32) > 0.0, 1.0, 0.0)

        # load ip-adapter
        ip_model = IPAdapter(pipe, image_encoder_ckpt, ip_ckpt, device)
        print_green("[INFO] Image-guided normal ControlNet is loaded.")

        # generate image variations
        images = ip_model.generate(pil_image=prompt, image=normal, num_samples=1, num_inference_steps=30, seed=seed)
        images[0] = Image.fromarray((np.array(images[0], dtype=np.float32) * mask).astype(np.uint8))
        images[0].save(f"{os.path.dirname(normal_path)}/ip.png")

    else:
        NotImplementedError("[INFO] Invalid ControlNet condition mode.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Employ ControlNet to generate the sample image.")
    parser.add_argument("--ctrlnet_seed", type=int, default=42)
    parser.add_argument("--controlnet_normal_path", type=str, default=None)
    parser.add_argument("--controlnet_cond_mode", type=str, default='text', choices=['text', 'image'])
    parser.add_argument("--prompt", type=str, default=None)

    args = parser.parse_args()

    generate_image_prompt_via_normal_controlnet(
        normal_path=args.controlnet_normal_path,
        prompt=args.prompt,
        controlnet_cond_mode=args.controlnet_cond_mode,
        seed=args.ctrlnet_seed,
    )
    