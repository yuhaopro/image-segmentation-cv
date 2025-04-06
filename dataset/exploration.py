import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="This script will generate the dataset directory structure. Run this script before training or testing.")
    parser.add_argument("--dir", "-d", dest="directory", help="Dataset directory eg. Dataset/TrainVal", required=True)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose mode")

    os.listdir()
    args = parser.parse_args()
    verbose = args.verbose
    dataset_dir = args.directory

    current_dir = os.getcwd()

    if dataset_dir[0] == '/':
        # append current directory path
        current_dir = ""
        # remove the '/'
        dataset_dir = dataset_dir[1:]

    dataset_mask_dir = f"{current_dir}/{dataset_dir}/label"
    plot_dog_to_cat_count(dataset_mask_dir)
    plot_categories_count(dataset_mask_dir)

def plot_categories_count(mask_dir):
    categories_count = get_categories_count(mask_dir)
    df = pd.DataFrame(list(categories_count.items()), columns=['Category', 'Count'])
    # Set plot style (optional)
    sns.set_style("whitegrid")

    # Create the horizontal bar plot using seaborn
    plt.figure(figsize=(8, 4)) # Optional: Adjust figure size
    # Assign numerical 'Count' to x and categorical 'Category' to y for horizontal plot
    barplot = sns.barplot(x='Count', y='Category', data=df, palette='viridis')

    # Add titles and labels (Seaborn uses DataFrame column names automatically for axes)
    plt.title("Counts per Category")

    # Optional: Add count labels on top of the bars
    # Get the current axes
    ax = plt.gca()
    # Add labels to each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

    plt.tight_layout() # Adjust layout
    plt.show()

def get_categories_count(mask_dir):
    masks = os.listdir(mask_dir)
    masks.remove("cats")
    masks.remove("dogs")
    category_counts = defaultdict(lambda: 0)
    
    for mask in masks:
        base_name, _ = os.path.splitext(mask)
        last_underscore_index = base_name.rfind('_')
        assert last_underscore_index != 0
        
        category_name = base_name[:last_underscore_index]
        category_counts[category_name] += 1
    return category_counts
    
def plot_dog_to_cat_count(mask_dir):
    cat_counts, dog_counts = get_dog_to_cat_count(mask_dir)
    data = {'Animal': ['Cats', 'Dogs'], 'Count': [cat_counts, dog_counts]}
    df = pd.DataFrame(data)

    # Set the style for the plot (optional)
    sns.set_style("whitegrid")

    # Create the bar plot
    plt.figure(figsize=(6, 4))
    barplot = sns.barplot(x='Animal', y='Count', data=df, palette='viridis')

    # Add titles and labels (optional but recommended)
    plt.title('Cats and Dogs Dataset Ratio')
    plt.xlabel('Animal Type')
    plt.ylabel('Count')

    # Add count labels on top of the bars (optional)
    for container in barplot.containers:
        barplot.bar_label(container, fmt='%d') # '%d' formats the label as an integer

    # Display the plot
    plt.show()
    
def get_dog_to_cat_count(mask_dir):
    cats_dir = f"{mask_dir}/cats"
    dogs_dir = f"{mask_dir}/dogs"
    
    cat_masks = os.listdir(cats_dir)
    dog_masks = os.listdir(dogs_dir)
    
    return (len(cat_masks), len(dog_masks))
    
    

if __name__ == "__main__":
    main()