import torch
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from PIL import Image
import numpy as np
import os

def run_vanilla_tests(masks,imageObjects):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "../stable-diffusion-2-base"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    num_inference_steps = 5 
    scheduler.set_timesteps(num_inference_steps)
    
    
    for image in imageObjects:
        output_dir = image.name
        os.makedirs(output_dir, exist_ok=True)
        for mask in masks:
            i = 1
            for prompt in image.prompts:
                prompt = pipe.encode_prompt(prompt, device, num_images_per_prompt=1, 
                                            do_classifier_free_guidance=False)[0]
                result_image = generate_image_vanilla(mask.image, image.image, prompt, pipe, 
                                                      scheduler, device, dtype)
                file_path = os.path.join(output_dir, 
                                         f"{mask.name}_mask-{image.name}-prompt{i}.png")
                result_image.save(file_path)
                i = i+1

def generate_image_vanilla(mask, image, prompt, pipe, scheduler, device, dtype):
    latents_x0, mask = preprocess(image, mask, pipe.vae, device, dtype)
    latents = torch.randn_like(latents_x0)
    
    for t in scheduler.timesteps:
        print (t)
        with torch.no_grad():
            noise_pred = pipe.unet(latents,t, encoder_hidden_states=prompt).sample

        latents_unknown = scheduler.step(noise_pred, t, latents).prev_sample

        noise = torch.randn_like(latents_x0)
        latents_known = scheduler.add_noise(latents_x0, noise, t-1)

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
    



