import os
import pandas as pd
from PIL import Image
import torch
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def run_evaluation(base_results_dir, original_images_dir):
    all_data = []

    for mask_type in os.listdir(base_results_dir):
        print (mask_type)
        mask_type_path = os.path.join(base_results_dir, mask_type)
        if not os.path.isdir(mask_type_path): continue

        for mask_name in os.listdir(mask_type_path):
            print (mask_name)

            mask_name_path = os.path.join(mask_type_path, mask_name)
            if not os.path.isdir(mask_name_path): continue

            for img_name in os.listdir(mask_name_path):
                gen_img_path = os.path.join(mask_name_path, img_name)
                orig_img_path = os.path.join(original_images_dir, img_name)

                if os.path.exists(orig_img_path):
                    orig_img = Image.open(orig_img_path).convert('RGB')
                    gen_img = Image.open(gen_img_path).convert('RGB')
                    
                    # חישוב מדדים
                    m = calculate_metrics(orig_img, gen_img)
                    
                    # שמירת המטא-דאטה
                    m.update({
                        'mask_type': mask_type,
                        'mask_name': mask_name,
                        'image_name': img_name
                    })
                    all_data.append(m)

    return pd.DataFrame(all_data)

def calculate_metrics(original_img, generated_img):
    def to_tensor(img):
        img = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
        return img.unsqueeze(0)

    orig = to_tensor(original_img)
    gen = to_tensor(generated_img)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    
    psnr_val = psnr_metric(gen, orig)
    ssim_val = ssim_metric(gen, orig)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    orig = orig.to(device)
    gen = gen.to(device)
    lpips_val = lpips_metric(gen, orig)

    return {
        "PSNR": psnr_val.item(),
        "SSIM": ssim_val.item(),
        "LPIPS": lpips_val.item()
    }
