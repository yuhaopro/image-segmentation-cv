import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint: str, model, device):
    print("=> Loading checkpoint")
    # original saved file with DataParallel
    checkpoint_info = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_info['state_dict'])
    
def plot_images_side_by_side(image1, image2, title1="Image 1", title2="Image 2", figure_title=None, figsize=(10, 5)):
    """
    Plots two images side-by-side using Matplotlib.

    The images should be in a format compatible with matplotlib's imshow,
    typically NumPy arrays with shape (H, W) for grayscale or (H, W, C) for color.
    Can also accept PIL Image objects.

    Args:
        image1: The first image data (e.g., NumPy array, PIL Image).
        image2: The second image data (e.g., NumPy array, PIL Image).
        title1 (str): Title for the first subplot (left image).
        title2 (str): Title for the second subplot (right image).
        figure_title (str, optional): Overall title for the figure. Defaults to None.
        figsize (tuple): Size of the overall figure (width, height in inches).
    """
    # --- Data Preparation (Optional: Convert PIL Images to NumPy arrays) ---
    # Matplotlib's imshow works best with NumPy arrays.
    # If PIL Image objects are passed, convert them.
    if Image and isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if Image and isinstance(image2, Image.Image):
        image2 = np.array(image2)

    # Basic check if inputs are numpy arrays now
    if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
        print("Warning: Inputs should preferably be NumPy arrays or PIL Images.")
        # Attempt to proceed, but imshow might fail if format is incompatible

    # --- Plotting ---
    # Create a figure containing 1 row and 2 columns of subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Display the first image on the left subplot (axes[0])
    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis('off')  # Hide the axes ticks and labels

    # Display the second image on the right subplot (axes[1])
    axes[1].imshow(image2)
    axes[1].set_title(title2)
    axes[1].axis('off')  # Hide the axes ticks and labels

    # Add an overall title to the figure if provided
    if figure_title:
        fig.suptitle(figure_title, fontsize=16)

    # Adjust layout to prevent titles from overlapping.
    # May need slight adjustment if using suptitle.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if figure_title else None)

    # Show the plot
    plt.show()