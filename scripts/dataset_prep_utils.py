import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
def main():
    # Example usage
    image_shape = (256, 256)  # Example image size
    coordinate = (120, 200)   # Example point
    heatmap = generate_coordinate_heatmap(image_shape, coordinate)
    
    # cv2.imwrite("heatmap.png", (heatmap * 255).astype(np.uint8))
    plot_heatmap(heatmap=heatmap)

def generate_ground_truth_mask(class_label, mask):
    # get the pixel value at this coordinate
    new_mask = np.zeros(np.shape(mask), dtype=np.uint8)
    new_mask[mask == class_label] = 255
    return new_mask
    
def save_mask_to_image(mask, output_path="mask.png"):
    """
    Saves a NumPy array mask (0 or 255) as a grayscale image.

    Args:
        mask (numpy.ndarray): The mask as a 2D NumPy array (0 or 255).
        output_path (str, optional): The path to save the image. Defaults to "mask.png".
    """

    image = Image.fromarray(mask, mode='L')  # 'L' for grayscale
    image.save(output_path)
    print(f"Mask saved to {output_path}")
    
    

def plot_heatmap(heatmap, title="Heatmap", cmap="viridis", interpolation="nearest", aspect="auto"):
    """Plots a 2D heatmap using matplotlib."""
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    plt.imshow(heatmap, cmap=cmap, interpolation=interpolation, aspect=aspect)
    plt.colorbar(label="Intensity")  # Add a colorbar for reference
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.tight_layout() # Ensures proper spacing
    plt.show()

def generate_coordinate_heatmap(image_shape, coordinate, sigma=10):
    """
    Generates a Gaussian heatmap for a given coordinate in an image.
    
    Args:
        image_shape (tuple): (H, W) of the image.
        coordinate (tuple): (x, y) location in the image.
        sigma (float): Standard deviation for Gaussian distribution.

    Returns:
        np.ndarray: Heatmap with shape (H, W), values normalized between 0 and 1.
    """
    H, W = image_shape
    x, y = coordinate

    # Create a meshgrid of coordinates
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    # Compute the Gaussian function
    # x and y are the "mean" of the gaussian, as it represents the peak
    heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    # Normalize between 0 and 1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap
def save_heatmap_to_image(heatmap, output_path="heatmap.png"):
    """
    Converts a NumPy array representing a Gaussian heatmap to a PIL Image and saves it.

    Args:
        heatmap (numpy.ndarray): The Gaussian heatmap as a 2D NumPy array.
        output_path (str, optional): The path to save the image. Defaults to "heatmap.png".
    """

    # Normalize the heatmap to the range [0, 255] for image representation.
    normalized_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255
    normalized_heatmap = normalized_heatmap.astype(np.uint8)

    # Create a PIL Image from the normalized heatmap.
    image = Image.fromarray(normalized_heatmap, mode='L')  # 'L' for grayscale

    # Save the image.
    image.save(output_path)
    print(f"Heatmap saved to {output_path}")
    
def generate_points_from_mask(mask, num_samples=1):
    """
    Sample points from the mask for each class.
    Args:
        mask: Segmentation mask with class labels (e.g., 1, 2, 3).
        num_samples: Number of points to sample for each class.
    
    Returns:
        A list of (x, y) points for each class.
    """
    height, width = mask.shape
    points = {0: [], 38: [], 75: []}

    # Iterate over the mask and collect points for each class
    for y in range(height):
        for x in range(width):
            class_label = mask[y, x]
            if class_label in points:
                points[class_label].append((x, y))
    
    sampled_points = []
    for class_label in points:
        if len(points[class_label]) != 0:
            random_coordinate = random.choice(points[class_label])
        
            heatmap = generate_coordinate_heatmap(mask.shape, random_coordinate, sigma=10)
            new_mask = generate_ground_truth_mask(class_label, mask)
            sampled_points.append((heatmap, new_mask))
    return sampled_points
        

if __name__ == "__main__":
    main()