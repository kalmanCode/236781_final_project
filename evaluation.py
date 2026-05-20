import os
import pandas as pd
from PIL import Image
import torch
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def run_evaluation(base_results_dir, original_images_dir):
    all_data = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for mask_type in os.listdir(base_results_dir):
        mask_type_path = os.path.join(base_results_dir, mask_type)
        if not os.path.isdir(mask_type_path): continue

        for mask_name in os.listdir(mask_type_path):
            mask_name_path = os.path.join(mask_type_path, mask_name)
            if not os.path.isdir(mask_name_path): continue

            for img_name in os.listdir(mask_name_path):
                if img_name.startswith('.') or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                gen_img_path = os.path.join(mask_name_path, img_name)
                orig_img_path = os.path.join(original_images_dir, img_name)

                if os.path.exists(orig_img_path):
                    orig_img = Image.open(orig_img_path).convert('RGB')
                    gen_img = Image.open(gen_img_path).convert('RGB')
                    
                    m = calculate_metrics(orig_img, gen_img, lpips_metric, psnr_metric, ssim_metric, device)
                    
                    m.update({
                        'mask_type': mask_type,
                        'mask_name': mask_name,
                        'image_name': img_name
                    })
                    all_data.append(m)
    
    # Clear cache to free up memory for the next experiment run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(all_data)

def calculate_metrics(original_img, generated_img, lpips_metric, psnr_metric, ssim_metric, device):
    def to_tensor(img):
        img = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
        return img.unsqueeze(0).to(device)

    orig = to_tensor(original_img)
    gen = to_tensor(generated_img)

    with torch.no_grad():
        psnr_val = psnr_metric(gen, orig)
        ssim_val = ssim_metric(gen, orig)
        lpips_val = lpips_metric(gen, orig)

    return {
        "PSNR": psnr_val.item(),
        "SSIM": ssim_val.item(),
        "LPIPS": lpips_val.item()
    }
    
import pandas as pd

def summarize_results(df_list, experiment_names):
    """
    Processes a list of DataFrames, adds experiment identifiers, and returns 
    a dictionary containing detailed average tables for each metric by image and by mask.
    """
    processed_dfs = []
    
    for df, name in zip(df_list, experiment_names):
        if df.empty:
            print(f"Warning: DataFrame for experiment '{name}' is empty. Skipping.")
            continue
            
        temp_df = df.copy()
        temp_df['exp'] = name
        
        # Save to an individual CSV file for each experiment
        temp_df.to_csv(f"results_{name}.csv", index=False)
        processed_dfs.append(temp_df)

    if not processed_dfs:
        raise ValueError("All provided DataFrames are empty. Cannot summarize.")

    # Consolidate all results
    all_results = pd.concat(processed_dfs, ignore_index=True)

    # Validate required columns
    required_cols = ['image_name', 'mask_name', 'exp', 'PSNR', 'SSIM', 'LPIPS']
    missing = [c for c in required_cols if c not in all_results.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrames: {missing}")

    # Calculate baseline averages
    metrics = ['PSNR', 'SSIM', 'LPIPS']
    image_means_raw = all_results.groupby(['image_name', 'exp'])[metrics].mean().reset_index()
    mask_means_raw = all_results.groupby(['mask_name', 'exp'])[metrics].mean().reset_index()

    # Create 6 comparison tables (Pivots)
    comparison_tables = {}
    
    for metric in metrics:
        # Table by image for the specific metric
        comparison_tables[f'{metric}_by_image'] = image_means_raw.pivot(
            index='image_name', columns='exp', values=metric
        )
        # Table by mask for the specific metric
        comparison_tables[f'{metric}_by_mask'] = mask_means_raw.pivot(
            index='mask_name', columns='exp', values=metric
        )

    # Return results organized neatly
    return {
        "all_results": all_results,
        "tables": comparison_tables,
        "raw_means": {
            "by_image": image_means_raw,
            "by_mask": mask_means_raw
        }
    }