import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 输出归一化到 [0,1] 之间
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 带分割头的自编码器类
class AutoencoderWithSegmentationHead(nn.Module):
    def __init__(self, encoder, decoder, num_classes=4):
        super(AutoencoderWithSegmentationHead, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),  # 根据编码器的输出调整
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),  # 生成分割预测
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)  # 上采样调整为256x256
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        segmentation_output = self.segmentation_head(encoded)
        return decoded, segmentation_output