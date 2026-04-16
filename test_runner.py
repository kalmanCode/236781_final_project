import torch
import os

def run_tests(masks,imageObjects, pipe,num_inference_steps=30):
    for image in imageObjects:
        output_dir = image.name
        os.makedirs(output_dir, exist_ok=True)
        for mask in masks:
            i = 1
            for prompt in image.prompts:
                result_image = generate_image(mask.image, image.image, prompt, pipe, num_inference_steps)
                file_path = os.path.join(output_dir, f"{mask.name}_mask-{image.name}-prompt{i}.png")
                result_image.save(file_path)
                i = i+1

def generate_image(mask_image,image,prompt,pipe,num_inference_steps):  
    result_image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps
    ).images[0]
    
    return result_image




