from PIL import Image
import os
import numpy as np

def get_num_of_each_class(dataset_dir):
    masks = os.listdir(dataset_dir)
    num_of_cats = 0
    num_of_dogs = 0

    for mask in masks:
        mask_img = np.array(
            Image.open(os.path.join(dataset_dir, mask)).convert(mode="L")
        )
        unique_mask_values = np.unique(mask_img)

        if 38 in unique_mask_values:
            num_of_cats += 1
        elif 75 in unique_mask_values:
            num_of_dogs += 1

    return num_of_cats, num_of_dogs

def main():
    # how dog vs cat pictures in dataset
    # use the mask unique values to determine
    # TRAIN_MASK_DIR = "/teamspace/studios/this_studio/image-segmentation-cv/Dataset/TrainVal/label"
    # TEST_MASK_DIR = "/teamspace/studios/this_studio/image-segmentation-cv/Dataset/Test/label"
    TRAIN_MASK_DIR = "Dataset/TrainVal/label"
    TEST_MASK_DIR = "Dataset/Test/label"
    num_of_cats, num_of_dogs = get_num_of_each_class(TRAIN_MASK_DIR)
    print(f"Train Dataset: Cats ({num_of_cats}), Dogs ({num_of_dogs})")
    print(f"Train Dataset Cats to Dogs ratio: {num_of_cats / num_of_dogs}")
    num_of_cats, num_of_dogs = get_num_of_each_class(TEST_MASK_DIR)
    print(f"Test Dataset: Cats ({num_of_cats}), Dogs ({num_of_dogs})")
    print(f"Test Dataset Cats to Dogs ratio: {num_of_cats / num_of_dogs}")

if __name__ == "__main__":
    main()