from typing import List
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

def plot_relationship(perturbation_name: str, perturbations: List[int], dice_scores: List[float]):
    # Create the plot
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.plot(perturbations, dice_scores, marker='o', linestyle='-', color='b', 
            linewidth=2, markersize=8, label='Dice Score')

    # Customize the plot
    plt.xlabel('Perturbation', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.title(f'Relationship Between {perturbation_name} and Dice Score', 
            fontsize=14, pad=10)
    plt.grid(True, linestyle='--', alpha=0.7)  # Add a subtle grid
    plt.xticks(perturbations)  # Show all perturbation values on x-axis
    plt.yticks(np.arange(0, 1.1, 0.1))  # Dice scores typically range from 0 to 1
    plt.legend(loc='lower left', fontsize=10)  # Add a legend

    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.savefig(f"{perturbation_name}_plot.png",format="png")

    # Display the plot
    plt.show()