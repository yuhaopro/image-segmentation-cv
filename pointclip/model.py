import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from transformers import CLIPVisionModel, CLIPImageProcessor
from torchvision.utils import save_image
from PIL import Image
import numpy as np


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
    
class ClipPointSeg(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_size=768, patch_size=32, image_size=224):
        super(ClipPointSeg, self).__init__()
        
        # Load pretrained CLIP vision model and processor
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Calculate grid size (e.g., 7x7 for 224x224 image with 32x32 patches)
        self.grid_size = image_size // patch_size
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_size*2, (hidden_size), kernel_size=2, stride=2),
            DoubleConv(hidden_size, hidden_size),
            nn.ConvTranspose2d(hidden_size, (hidden_size // 2), kernel_size=2, stride=2),
            DoubleConv((hidden_size // 2),(hidden_size // 2)),
            nn.ConvTranspose2d((hidden_size // 2), (hidden_size // 4), kernel_size=2, stride=2),
            DoubleConv((hidden_size // 4),(hidden_size // 4)),
            nn.ConvTranspose2d((hidden_size // 4), (hidden_size // 8), kernel_size=2, stride=2),
            DoubleConv((hidden_size // 8),(hidden_size // 8)),
        )

        self.final_conv = nn.Conv2d((hidden_size // 8), out_channels, kernel_size=1)
        
        
    def forward(self, image, point):
        # x is a batch of images: [B, C, H, W]
        batch_size = image.shape[0]
        device = image.device
        print(f"Image Shape:{image.shape}, Point Shape: {point.shape}")
        # process image through CLIP
        inputs_image = self.processor(images=image, return_tensors="pt")
        inputs_image = {k: v.to(device) for k, v in inputs_image.items()}
        outputs_image = self.model(**inputs_image).last_hidden_state  # [B, S, HS]
        features_image = outputs_image[:, 1:, :]  # [B, S-1, HS]
        # reshape encoding to spatial feature for image
        # For clip-vit-base-patch32 with 224x224 input: S-1=49 (7x7 grid)
        features_image = features_image.permute(0, 2, 1)  # [B, HS, S-1]
        features_image = features_image.view(batch_size, -1, self.grid_size, self.grid_size)  # [B, HS, 7, 7]
        

        # process heatmap through CLIP
        inputs_point = self.processor(images=point, return_tensors="pt")
        inputs_point = {k: v.to(device) for k, v in inputs_image.items()}
        outputs_point = self.model(**inputs_point).last_hidden_state  # [B, S, HS]
        features_point = outputs_point[:, 1:, :]  # [B, S-1, HS]
        features_point = features_point.permute(0, 2, 1)  # [B, HS, S-1]
        features_point = features_point.view(batch_size, -1, self.grid_size, self.grid_size)        
        
        features = torch.cat((features_image, features_point), dim=1) # new HS = 768 * 2 = 1536
        # Decode features
        features = self.decoder(features)  # [B, 128, 7, 7]
        # print(f"Decode features: {features.shape}")

        # final_conv
        seg_map = self.final_conv(features)  # [B, out_channels, 28, 28]
        # print(f"final_conv: {seg_map.shape}")
        
        # Final upsampling to input resolution
        seg_map = TF.resize(seg_map, size=image.shape[2:], interpolation=TF.InterpolationMode.BILINEAR)
        # print(f"Final upsampling to input resolution: {seg_map.shape}")
       
        return seg_map

def test():
    model = ClipPointSeg(out_channels=1)  # binary for point segmentation
    image = np.array(Image.open("images/Abyssinian_1_color.jpg"))
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.permute(2,0,1)
    image_tensor = image_tensor.unsqueeze(0)
    assert image_tensor.size() == (1, 3, 400, 600) 
    
    point = np.array(Image.open("images/Abyssinian_1_point_696.png"))
    point = point[:, :, np.newaxis]
    point = np.tile(point, (1,1,3))
    point_tensor = torch.from_numpy(point)
    point_tensor = point_tensor.permute(2,0,1)
    point_tensor = point_tensor.unsqueeze(0)

    assert point_tensor.size() == (1, 3, 400, 600) 

    output = model(image_tensor, point_tensor)
    assert output.size() == (1, 1, 400, 600)
    
    mask = np.array(Image.open("images/Abyssinian_1_point_696_mask.png"))
    mask_tensor = torch.from_numpy(mask)
    assert mask_tensor.size() == (400,600)
    
if __name__ == "__main__":
    test()