import torch

# create proper class labels
class_to_color = {0: 0, 1: 38, 2: 75, 3: 0}  # Background  # Cat  # Dog  # Boundary

"""
Conversion of mask pixel to class labels.
Boundary is counted as background.
"""
color_to_class = {0: 0, 38: 1, 75: 2, 255: 0}  # Background  # Cat  # Dog  # Boundary

color_to_class_test = {
    0: 0,  # Background
    38: 1,  # Cat
    75: 2,  # Dog
    255: 3,  # Boundary
}

color_to_class_heatmap = {
    0: 0,
    255: 1,
}


def convert_class_to_color(class_mask, class_to_color):
    placeholder = torch.zeros_like(class_mask)
    for class_idx, color in class_to_color.items():
        placeholder[class_mask == class_idx] = color
    return placeholder


def convert_color_to_class(color_mask, color_to_class):
    placeholder = torch.zeros_like(color_mask)
    for color_idx, _class in color_to_class.items():
        placeholder[color_mask == color_idx] = _class
    return placeholder


def remove_class_dimension(mask):
    class_indices = torch.argmax(mask, dim=0)
    return class_indices
