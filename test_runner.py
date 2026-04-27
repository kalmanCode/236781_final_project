import torch
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from PIL import Image
import numpy as np
import os

def run_vanilla_tests(masks, imageObjects, cfg=0, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "../stable-diffusion-2-base"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)
    
    
    for image in imageObjects:
        output_dir = image.name
        os.makedirs(output_dir, exist_ok=True)
        for mask in masks:
            i = 1
            for prompt in image.prompts:
                if seed > 0:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                do_cfg = cfg > 0
                prompt_embeds, negative_embeds = pipe.encode_prompt(prompt, device, num_images_per_prompt=1, 
                                            do_classifier_free_guidance=do_cfg)
                if do_cfg:
                    prompt_input = torch.cat([negative_embeds, prompt_embeds], dim=0)
                else:
                    prompt_input = prompt_embeds
                result_image = generate_image_vanilla(mask.image, image.image, prompt_input, pipe, 
                                                      scheduler, device, dtype, cfg)
                file_path = os.path.join(output_dir, 
                                         f"{mask.name}_mask-{image.name}-prompt{i}.png")
                result_image.save(file_path)
                i = i+1

def generate_image_vanilla(mask, image, prompt, pipe, scheduler, device, dtype, cfg):
    latents_x0, mask = preprocess(image, mask, pipe.vae, device, dtype)
    latents = torch.randn_like(latents_x0)
    
    for i, t in enumerate(scheduler.timesteps):
        t = t.item()
        if cfg > 0:
            latents_model = torch.cat([latents] * 2)
        else:
            latents_model = latents

        t_input = torch.tensor([t] * latents_model.shape[0], device=device).to(dtype=torch.long)
        with torch.no_grad():
            noise_pred = pipe.unet(latents_model,t_input, encoder_hidden_states=prompt).sample

        if cfg > 0:
            noise_pred_no_prompt, noise_pred_prompt = noise_pred.chunk(2)
            noise_pred = noise_pred_no_prompt + cfg * (noise_pred_prompt - noise_pred_no_prompt)

        if i < len(scheduler.timesteps) - 1:
            t_prev = scheduler.timesteps[i+1].item()
        else:
            t_prev = 0

        latents_unknown = step(scheduler, noise_pred, latents, t, t_prev)
                
        if i < len(scheduler.timesteps) - 1:
            t_prev = scheduler.timesteps[i+1].item()
        else:
            t_prev = 0
        latents_known = add_noise(scheduler, latents_x0, t_prev)

        latents = (mask * latents_known) + ((1.0 - mask) * latents_unknown)
    
    result_image = postprocess (latents, pipe.vae)
    return result_image

def preprocess(image, mask, vae, device, dtype):
    img_arr = np.array(image.resize((512, 512))).astype(np.float32) / 127.5 - 1.0
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)    

    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample()
        latents = latents * 0.18215
    
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = 1.0 - mask 
    mask = torch.from_numpy(mask).to(device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    
    return latents, mask

def postprocess (latents, vae):
    latents = latents / 0.18215
    with torch.no_grad():
        image = vae.decode(latents).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).astype(np.uint8)
    
    return Image.fromarray(image)
    
def step(scheduler, pred_noise, latent, t, t_prev):
    
    alpha_bar_t = scheduler.alphas_cumprod[t]
    alpha_bar_t_prev = scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(latent.device)

    alpha_t = alpha_bar_t / alpha_bar_t_prev
    beta_t = 1 - alpha_t

    # mu = (1 / sqrt(alpha_t)) * (latent - (beta_t / sqrt(1 - alpha_bar_t)) * pred_noise)
    
    coeff_inv_sqrt_alpha = 1 / (alpha_t ** 0.5)
    coeff_noise = beta_t / ((1 - alpha_bar_t) ** 0.5)
    
    mu = coeff_inv_sqrt_alpha * (latent - coeff_noise * pred_noise)

    if t > 0:
        variance = ( (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) ) * beta_t
        noise = torch.randn_like(latent)
        return mu + (variance ** 0.5) * noise
    
    return mu

def add_noise(scheduler, latents, t):
    if t < 0:
        return latents
    noise = torch.randn_like(latents)
    alpha_prod = scheduler.alphas_cumprod[t]
    return (alpha_prod ** 0.5) * latents + ((1-alpha_prod)** 0.5) * noise


