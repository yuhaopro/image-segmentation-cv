import torch
import torch.nn as nn
from torchview import draw_graph
import torchvision.transforms.functional as TF 
import matplotlib.pyplot as plt
import os 
SAVE_DIRECTORY = f"{os.getcwd()}/unet/"

"""
DoubleConv is used for both the encoding and decoding processes, and should be able to accomodate the change in channels at each stage during downsampling and upsampling.
"""
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding="valid"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv(x)

"""
UNET class consists of the entire segmentation network. 
The components are:
- double convolutional layers
- max pooling 
- concatenation
- upsampling
- final convolution layer 
"""
class UNET(nn.Module):
    def __init__(
            # RGB images will have 3 channels
            # number of convolution kernels based off UNET architecture paper
            self, in_channels=3, out_channels=3, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()

        # uses module list to store layers to be iterated
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) # eg input RGB image has 3 channels, 64 output channels)
            in_channels = feature # new input channel will be 64

        # Up part of UNET
        # channels get reduced during upsampling to merge with skip connections
        for feature in reversed(features):
            self.ups.append(
                # channel reduction
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            
            # after concat with skip connections -> 512 + 512 = 1024, double conv needs input features*2
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # out channels should correspond to num of classes
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=(1,1), stride=1, padding="valid")
        
    def forward(self, x):
        
        image_shape = x.shape[2:]
        # [64, 128, 256, 512]
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # reverse the skip connections
        skip_connections = skip_connections[::-1]

        # step 2 because up contains upsample and conv
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        x = self.final_conv(x)
        return TF.resize(x, size=image_shape, interpolation=TF.InterpolationMode.BILINEAR)

def test():
    x = torch.randn((3, 3, 256, 256))
    model = UNET(in_channels=3, out_channels=3)
    
    draw_graph(model=model, input_data=x, save_graph=True, filename="UNET_architecture", directory=SAVE_DIRECTORY)


if __name__ == "__main__":
    test()