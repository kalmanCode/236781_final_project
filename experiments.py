from masks import create_checkerboard_mask, create_vertical_stripe_mask, create_horizontal_stripe_mask, create_stochastic_mask
from InpaintGenerator import VanillaGenerator, ResamplingGenerator
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from PIL import Image
import numpy as np
import os

def create_masks(seed, show):
    masks = []
    for k in range(0,5):
        mask = create_checkerboard_mask(2**k)
        if show:
            print(mask.name)
            plt.imshow(mask.image, cmap='gray')
            plt.axis('off') # This removes the axis numbers
            plt.show()
        masks.append(mask)
    
    for k in range(0,4):
        mask = create_vertical_stripe_mask(2**k)
        if show:
            print(mask.name)
            plt.imshow(mask.image, cmap='gray')
            plt.axis('off') # This removes the axis numbers
            plt.show()
        masks.append(mask)
    
    for k in range(0,4):
        mask = create_horizontal_stripe_mask(2**k)
        if show:
            print(mask.name)
            plt.imshow(mask.image, cmap='gray')
            plt.axis('off') # This removes the axis numbers
            plt.show()
        masks.append(mask)

    for i in range(0,3):
        mask = create_stochastic_mask(0.5, seed = seed+i)
        mask.name = f"{mask.name}_i={i}"
        if show:
            print(mask.name)
            plt.imshow(mask.image, cmap='gray')
            plt.axis('off') # This removes the axis numbers
            plt.show()
        masks.append(mask)
        
    return masks


def create_pipe(num_inference_steps=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "../stable-diffusion-2-base"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps)

    return pipe, scheduler, device

def vanilla_experiment(images, seed=42, show=False, cfg=0):
    masks = create_masks(seed, show)
    pipe, scheduler, device = create_pipe(50)
    vanilla_generator = VanillaGenerator()

    for mask in masks:
        do_cfg = cfg > 0
        if do_cfg:
            cfg_str = f"_cfg={cfg}"
        else:
           cfg_str = ""

        output_dir = f"vanilla{cfg_str}/{mask.type}/{mask.name}"
        os.makedirs(output_dir, exist_ok=True)
        if show:
            print(f"{mask.name}")
        for image in images:
            if seed > 0:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            prompt_embeds, negative_embeds = pipe.encode_prompt(image.prompt, device, num_images_per_prompt=1, 
                                                                do_classifier_free_guidance=do_cfg)
            if do_cfg:
                prompt_input = torch.cat([negative_embeds, prompt_embeds], dim=0)
            else:
                prompt_input = prompt_embeds
                
            result_image = vanilla_generator.generate(mask.image,image.image, prompt_input ,pipe,scheduler,cfg)
            file_path = os.path.join(output_dir, f"{image.name}.png")
            result_image.save(file_path)
            if show:
                print (f"created {image.name} with mask {mask.name}")


def resampling_experiment(images, seed=42, show=False, cfg=0, jump_length=10, jump_n_sample=10):
    masks = create_masks(seed, show)
    pipe, scheduler, device = create_pipe(50)
    resampling_generator = ResamplingGenerator(jump_length, jump_n_sample)

    for mask in masks:
        do_cfg = cfg > 0
        if do_cfg:
            cfg_str = f"_cfg={cfg}"
        else:
           cfg_str = ""
        output_dir = f"resampling{cfg_str}_jump_length_{jump_length}_jump_n_samples_{jump_n_sample}/{mask.type}/{mask.name}"
        os.makedirs(output_dir, exist_ok=True)
        if show:
            print(f"{mask.name}:")
        for image in images:
            if seed > 0:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            prompt_embeds, negative_embeds = pipe.encode_prompt(image.prompt, device, num_images_per_prompt=1, 
                                                                do_classifier_free_guidance=do_cfg)
            if do_cfg:
                prompt_input = torch.cat([negative_embeds, prompt_embeds], dim=0)
            else:
                prompt_input = prompt_embeds
                
            result_image = resampling_generator.generate(mask.image,image.image, prompt_input ,pipe,scheduler,cfg)
            file_path = os.path.join(output_dir, f"{image.name}.png")
            result_image.save(file_path)
            if show:
                print (f"created {image.name} with mask {mask.name}")


