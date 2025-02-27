import torch
import pickle as pkl
import torch.nn as nn
import clip
from PIL import Image

CLIP_MODEL = "ViT-B/32"
NUM_CLASSES = 3
DEVICE = 'cpu'

# adapted from the open source CLIPSeg github
class CLIPSegmentationModel(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", num_classes=3, device='cpu'):
        super(CLIPSegmentationModel, self).__init__()
        
        # Load CLIP vision encoder
        self.clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
        self.vision_model = self.clip_model.visual

        # Freeze CLIP parameters
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),  # CLIP ViT-B/32 output dim is 768
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1),  # Output: num_classes channels
        )

    def forward(self, x):
        # Forward pass through CLIP vision encoder
        with torch.no_grad():
            visual_features = self.vision_model(x)  # Shape: (B, 768)

        # Reshape visual features to 2D spatial format
        batch_size, hidden_size = visual_features.shape

        # preprocess transformer output of clip 
        height = width = int((visual_features.shape[1] - 1) ** 0.5)  # Exclude CLS token 
        visual_features = visual_features[:, 1:, :]  # Remove CLS token (B, 49, 768)
        visual_features = visual_features.permute(0, 2, 1).view(batch_size, hidden_size, height, width)  # Reshape to (B, 768, H, W)

        # Forward pass through segmentation head
        output = self.segmentation_head(visual_features)  # Shape: (B, num_classes, H, W)

        return output

def main():
    # list available clip models
    print(clip.available_models())
    _, preprocessor = clip.load(CLIP_MODEL, device=DEVICE, jit=False)

    # get an example image
    image = Image.open("Dataset/Test/color/Abyssinian_2.jpg").convert(mode="RGB")
    image_input = preprocessor(image).unsqueeze(0).to(DEVICE)
    print(f"image_input shape: {image_input.shape}")
    # create the model
    model = CLIPSegmentationModel(clip_model_name=CLIP_MODEL, num_classes=NUM_CLASSES, device=DEVICE)
    output = model(image_input)
if __name__ == "__main__":
    main()






















# processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
# model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# texts = ["a cat"]
# image = train_dataset[:1]["image"]
# inputs = processor(text=texts, images=image * len(texts), padding=True, return_tensors="pt")

# outputs = model(**inputs)

# logits = outputs.logits
# print(logits.shape)
# print(logits)

# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# # Convert logits to a numpy array
# seg_maps = logits.detach().cpu().numpy()  # Shape: (3, 352, 352)

# # Normalize each segmentation map separately
# for i, seg_map in enumerate(seg_maps):
#     seg_map = (seg_map - seg_map.min()) / (seg_map.max() - seg_map.min())  # Normalize to [0, 1]
#     seg_map = (seg_map * 255).astype(np.uint8)  # Scale to [0, 255]

#     # Display the segmentation map
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1,2,1)
#     plt.imshow(seg_map, cmap="gray")
#     plt.axis("off")
#     plt.title("prediction")

#     plt.subplot(1,2,2)
#     plt.imshow(train_dataset[:1]["mask"][0].resize((224,224)), cmap="gray")
#     plt.axis("off")
#     plt.title("target")

# plt.show()
