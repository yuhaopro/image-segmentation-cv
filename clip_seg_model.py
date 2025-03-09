import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from transformers import CLIPVisionModel, CLIPImageProcessor
from torchvision.utils import save_image
from PIL import Image


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv(x)
    
class ClipSegmentation(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_size=768, patch_size=32, image_size=224):
        super(ClipSegmentation, self).__init__()
        
        # Load pretrained CLIP vision model and processor
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Calculate grid size (e.g., 7x7 for 224x224 image with 32x32 patches)
        self.grid_size = image_size // patch_size
        
        # Projection layer to reshape features
        self.proj = nn.Conv2d(hidden_size, 512, kernel_size=1)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            DoubleConv(512,512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            DoubleConv(256,256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            DoubleConv(128,128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            DoubleConv(64,64),
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        
    def forward(self, x):
        # x is a batch of images: [B, C, H, W]
        batch_size = x.shape[0]
        
        # Process input through CLIP processor
        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(x.device) for k, v in inputs.items()}
        
        # Get CLIP vision features
        outputs = self.model(**inputs).last_hidden_state  # [B, S, HS]
        # print(f"Last Hidden State: {outputs.shape}")
        # Remove CLS token (first token) and reshape
        features = outputs[:, 1:, :]  # [B, S-1, HS]
        # print(f"Remove CLS token: {features.shape}")

        # Reshape to spatial grid
        # For clip-vit-base-patch32 with 224x224 input: S-1=49 (7x7 grid)
        features = features.permute(0, 2, 1)  # [B, HS, S-1]
        #print(f"Shift HS to Channels dimension: {features.shape}")

        features = features.view(batch_size, -1, self.grid_size, self.grid_size)  # [B, HS, 7, 7]
        # print(f"Reshape to spatial grid: {features.shape}")

        # Project features
        features = self.proj(features)  # [B, 512, 7, 7]
        # print(f"Project features: {features.shape}")

        # Decode features
        features = self.decoder(features)  # [B, 128, 7, 7]
        # print(f"Decode features: {features.shape}")

        # final_conv
        seg_map = self.final_conv(features)  # [B, out_channels, 28, 28]
        # print(f"final_conv: {seg_map.shape}")
        
        # Final upsampling to input resolution
        seg_map = TF.resize(seg_map, size=x.shape[2:], interpolation=TF.InterpolationMode.BILINEAR)
        # print(f"Final upsampling to input resolution: {seg_map.shape}")
       
        return seg_map

def main():
    model = ClipSegmentation(out_channels=3)  # 3 classes for segmentation
    image = read_image("images/Abyssinian_1_color.jpg")
    image_tensor = image.unsqueeze(0)
    print(f"Input shape: {image_tensor.size()}")  # Should be [2, 2, 224, 224]

    output = model(image_tensor)
    print(f"Output shape: {output.size()}")  # Should be [2, 2, 224, 224]
    seg_mask = output.squeeze(0)
    save_image(seg_mask, "clip_seg_test.png")
if __name__ == "__main__":
    main()