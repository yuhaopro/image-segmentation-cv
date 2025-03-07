from PIL import Image
import os
import numpy as np
import shutil

"""
RUN THIS SCRIPT BEFORE RUNNING any *train.py files!
This python script will automatically split the original provided dataset into separate cats and dogs folder.
"""

TRAIN_COLOR_DIR = "Dataset/TrainVal/color"
TRAIN_MASK_DIR = "Dataset/TrainVal/label"
os.system(f"mkdir {os.getcwd()}/{TRAIN_COLOR_DIR}/cats")
os.system(f"mkdir {os.getcwd()}/{TRAIN_MASK_DIR}/cats")
os.system(f"mkdir {os.getcwd()}/{TRAIN_COLOR_DIR}/dogs")
os.system(f"mkdir {os.getcwd()}/{TRAIN_MASK_DIR}/dogs")

def split_dataset(mask_dir=TRAIN_MASK_DIR):
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
            shutil.move(f"{TRAIN_COLOR_DIR}/{image_name}", f"{TRAIN_COLOR_DIR}/cats/{image_name}")
            shutil.move(f"{TRAIN_MASK_DIR}/{mask_name}", f"{TRAIN_MASK_DIR}/cats/{mask_name}")
            
        # is dog
        if 75 in unique_mask_values:
            # print(f"is a dog.")
            shutil.move(f"{TRAIN_COLOR_DIR}/{image_name}", f"{TRAIN_COLOR_DIR}/dogs/{image_name}")
            shutil.move(f"{TRAIN_MASK_DIR}/{mask_name}", f"{TRAIN_MASK_DIR}/dogs/{mask_name}")

def main():
    split_dataset(TRAIN_MASK_DIR)

if __name__ == "__main__":
    main()