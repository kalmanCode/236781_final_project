from PIL import Image
import re
import torch
import os
import json
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageObject:
    def __init__(self, name, image, prompt):
        self.name = name
        self.image = image
        self.prompt = prompt

def get_concise_name(caption):
    ignore_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'with', 'by', 'is', 'and', 'sitting', 'standing', 'photo', 'image', 'background', 'there'}
    words = re.sub(r'[^a-z0-9\s]', '', caption.lower()).split()
    keywords = [w for w in words if w not in ignore_words]
    return "_".join(keywords[:3]) if keywords else "image_object"

def prepare_research_data(input_folder, output_json="images_metadata.json"):
    """
    Generate prompt from image and save metadata into JSON file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading BLIP model for automatic prompting...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_safetensors=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", use_safetensors=True).to(device)
    metadata = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        raw_image = Image.open(img_path).convert('RGB')
        
        # step 1: automatic prompting
        inputs = processor(raw_image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        
        # step 2: building metadata
        obj_name = get_concise_name(caption)
        entry = {
            "name": obj_name,
            "filename": filename,
            "prompt": caption,
        }
        metadata.append(entry)
        print(f"Processed: {filename} -> {obj_name}")

    # save to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    print(f"Metadata saved to {output_json}")

def load_objects_from_json(json_path, images_folder):
    """
    Load the json file and create list of 'ImageObject's
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    image_objects = []
    for entry in metadata:
        img_path = os.path.join(images_folder, entry["filename"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            obj = ImageObject(
                name=entry["name"],
                image=img,
                prompt=entry["prompt"],
            )
            image_objects.append(obj)
    
    return image_objects

def resize_images(input_dir, output_dir, size=(512, 512)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            with Image.open(img_path) as img:
                # שמירה על יחס גובה-רוחב באמצעות Center Crop
                width, height = img.size
                new_side = min(width, height)
                left = (width - new_side) / 2
                top = (height - new_side) / 2
                right = (width + new_side) / 2
                bottom = (height + new_side) / 2
                
                img = img.crop((left, top, right, bottom))
                img = img.resize(size, Image.Resampling.LANCZOS)
                
                img.save(os.path.join(output_dir, filename))
                print(f"Processed: {filename}")


