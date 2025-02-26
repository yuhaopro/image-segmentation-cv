import os
from datasets import Dataset, Features, Image
import numpy as np
import albumentations as A
import PIL as pil
import matplotlib.pyplot as plt
import pickle as pkl
from albumentations.pytorch import ToTensorV2

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

# imports the data into a dictionary format
def data_to_dict(split_path):
    images = []
    masks = []
    color_dir = os.path.join(split_path, 'color')
    label_dir = os.path.join(split_path, 'label')
    
    # Sort to ensure the pairing order is consistent
    for fname in sorted(os.listdir(color_dir)):
        image_path = os.path.join(color_dir, fname)
        mask_fname = fname.replace("jpg", "png")
        mask_path = os.path.join(label_dir, mask_fname)
        
        images.append(image_path)
        masks.append(mask_path)
        
    return {"image": images, "mask": masks} 

# perform dataset augmentation
def get_apply_transform_fn(augmentation):
    def apply_transform(data):
        image = np.array(data["image"], dtype=np.uint8)
        mask = np.array(data["mask"], dtype=np.uint8)
        augmented = augmentation(image=image, mask=mask)
        data["image"] = augmented["image"]
        data["mask"] = augmented["mask"]
        return data
    return apply_transform


# visualize the transform images
def visualize_transform_images(max, dataset):
    plt.figure(figsize=(10, 4))
    for idx in range(max):
        plt.subplot(2,5,idx+1)
        fig = plt.imshow(dataset[idx]["image"][idx].permute(1, 2, 0))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    for idx in range(max):
        plt.subplot(2,5,idx+1+5)
        fig = plt.imshow(dataset[idx]["mask"][idx].permute(1, 2, 0))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    
    plt.show()


def main():
    print(f"Loading the dataset into a dictionary...")
    train_val_data = data_to_dict("Dataset/TrainVal")
    test_data = data_to_dict("Dataset/Test")

    # features for creating Dataset object
    features = Features({
        "image": Image(mode="RGB"),
        "mask": Image(mode="RGB")
    })
    print(f"Creating the Dataset objects...")
    train_val_dataset = Dataset.from_dict(train_val_data, features=features)
    test_dataset = Dataset.from_dict(test_data, features=features)


    # augmentation parameters
    transform = A.Compose([
        A.Resize(height=IMAGE_WIDTH, width=IMAGE_HEIGHT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),               
        A.Rotate(limit=(-40,40)),
        A.ColorJitter(p=0.5), # randomly changes the brightness, contrast, saturation, and hue of an image
    ], seed=137)

    print(f"Augmenting the dataset...")

    aug_transform_fn = get_apply_transform_fn(transform)
    aug_train_val_dataset = train_val_dataset.map(aug_transform_fn)
    
    print(f"Splitting the dataset into training and validation...")

    print(f"Visualizing the first 5 transformed images...")
    visualize_transform_images(5, aug_train_val_dataset)

    train_val_dataset = aug_train_val_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_val_dataset["train"]
    val_dataset = train_val_dataset["test"]

    print(f"Visualizing the first 5 transformed images...")
    visualize_transform_images(5, train_dataset)

    

    print(f"Saving the augmented datasets...")

    # save the dataset into a pickle file to be loaded in
    with open("processed_dataset/train.pkl", "wb") as f:
        pkl.dump(train_dataset, f)

    with open("processed_dataset/val.pkl", "wb") as f:
        pkl.dump(val_dataset, f)

    with open("processed_dataset/test.pkl", "wb") as f:
        pkl.dump(test_dataset, f)

if __name__ == "__main__":
    main()