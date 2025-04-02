from PIL import Image
import os
import numpy as np
import shutil
import argparse
from dataset_prep_utils import generate_points_from_mask, save_heatmap_to_image, save_mask_to_image
"""
This script should be ran before the any training or testing.
"""
def main():
    parser = argparse.ArgumentParser(description="This script will generate the dataset directory structure. Run this script before training or testing.")
    parser.add_argument("--dir", "-d", dest="directory", help="Dataset directory eg. Dataset/TrainVal", required=True)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose mode")

    args = parser.parse_args()
    verbose = args.verbose
    dataset_dir = args.directory
    
    current_dir = os.getcwd()

    if dataset_dir[0] == '/':
        # append current directory path
        current_dir = ""
        # remove the '/'
        dataset_dir = dataset_dir[1:]
    dataset_color_dir = f"{current_dir}/{dataset_dir}/color"
    dataset_mask_dir = f"{current_dir}/{dataset_dir}/label"
    dataset_heatmap_dir = f"{current_dir}/{dataset_dir}/heatmap"

    if not (os.path.exists(dataset_color_dir) and os.path.exists(dataset_mask_dir)):
        print(f"Your dataset directory is missing the color or label directory!")
        return
    try:
        os.mkdir(f"{dataset_color_dir}/cats")
    except Exception as e:
        print(e)
    try:
        os.mkdir(f"{dataset_mask_dir}/cats")
    except Exception as e:
        print(e)
    try:
        os.mkdir(f"{dataset_color_dir}/dogs")
    except Exception as e:
        print(e)
    try:
        os.mkdir(f"{dataset_mask_dir}/dogs")
    except Exception as e:
        print(e)
    try:
        os.mkdir(dataset_heatmap_dir)
    except Exception as e:
        print(e)
    
    
    split_dataset(dataset_mask_dir, dataset_color_dir)
    generate_heatmap_dataset(mask_dir=dataset_mask_dir ,heatmap_dir=dataset_heatmap_dir, animal_class="cats")
    generate_heatmap_dataset(mask_dir=dataset_mask_dir, heatmap_dir=dataset_heatmap_dir, animal_class="dogs")

def split_dataset(mask_dir, color_dir):
    # color is in jpg, while mask is in png

    mask_names = os.listdir(mask_dir)
    mask_names.remove("cats")
    mask_names.remove("dogs")


    for mask_name in mask_names:
        mask = np.array(Image.open(f"{mask_dir}/{mask_name}").convert(mode="L"))
        unique_mask_values = np.unique(mask)
        image_name = mask_name.replace("png", "jpg")
        
        # is cat
        if 38 in unique_mask_values:
            # print(f"is a cat.")
            # shift the image to new directory label
            shutil.copy2(f"{color_dir}/{image_name}", f"{color_dir}/cats/{image_name}")
            shutil.copy2(f"{mask_dir}/{mask_name}", f"{mask_dir}/cats/{mask_name}")
            
        # is dog
        if 75 in unique_mask_values:
            # print(f"is a dog.")
            shutil.copy2(f"{color_dir}/{image_name}", f"{color_dir}/dogs/{image_name}")
            shutil.copy2(f"{mask_dir}/{mask_name}", f"{mask_dir}/dogs/{mask_name}")

def generate_heatmap_dataset(mask_dir, heatmap_dir, animal_class="cats"):
    mask_names = os.listdir(f"{mask_dir}/{animal_class}")
    
    point_dir = f"{heatmap_dir}/points"
    point_mask_dir = f"{heatmap_dir}/masks"
    try:
        os.mkdir(point_dir)
    except Exception as e:
        print(e)
    try:
        os.mkdir(point_mask_dir)
    except Exception as e:
        print(e)
    
    counter = 0
    for idx, mask_name in enumerate(mask_names):
        mask = np.array(Image.open(f"{mask_dir}/{animal_class}/{mask_name}").convert(mode="L"))
        filename = mask_name.split(".")[0]
        samples = generate_points_from_mask(mask=mask, num_samples=1)
        for (heatmap, new_mask) in samples:
            os.chdir(point_dir)
            heatmap_filename = f"{filename}_point_{counter}.png"
            save_heatmap_to_image(heatmap=heatmap, output_path=heatmap_filename)
            
            os.chdir(point_mask_dir)
            save_mask_to_image(mask=new_mask, output_path=heatmap_filename)
            counter += 1
            

if __name__ == "__main__":
    main()