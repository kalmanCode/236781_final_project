README

in repository 'images' - the original images (before resizing).

in 0-prepare_images.ipynb: we resize the images to 512 on 512 and save them in project images, then automatic prompting using BLIP and save the metadata to JSON file

in 1-inpaiting.ipynb: we load the objects from the json file, and save the images with their new name in repository 'base'

we run vanilla_experiment, and 6 resampling_experiment. To run the experiment, the parent repository must contain 'stable-diffusion-2-base'.

in 2-evaulation.ipynb: we calculate ssim, psnr and lpips metrics for each model results, summarize the data, save all the results to csv files and show in tables the performence of each model in each metric. 

notebooks:
0-prepare_images.ipynb
1-inpaint.ipynb
2-evaulation.ipynb

python files:
prepare_images.py
experiments.py
masks.py
InpaintGenerator.py
evaulation.py

other files:
images/
(i'm not sure about:
project_images/
base/
images_metadate.json
csv files [i think we should, maybe in addition]
and result repos 
)
