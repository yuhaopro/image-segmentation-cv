import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            # 第一层卷积，输入通道3，输出64，卷积核3x3，步长1，填充1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样

            # 第二层卷积，输入通道64，输出128，卷积核3x3，步长1，填充1
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样

            # 第三层卷积，输入通道128，输出256，卷积核3x3，步长1，填充1
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样

            # 第四层卷积，输入通道256，输出512，卷积核3x3，步长1，填充1
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样

            # 最后输出一个flatten的特征向量，大小可以根据实际需求设定
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, latent_dim)  # 假设图像尺寸从128x128经过四次下采样变为8x8
        )

    def forward(self, x):
        return self.encoder(x)

# 测试自编码器编码部分
def test_encoder():
    x = torch.randn(1, 3, 128, 128)  # 假设输入为1张3通道128x128的图像
    encoder = Encoder(in_channels=3, latent_dim=256)
    encoded = encoder(x)
    print(f"Encoded shape: {encoded.shape}")

if __name__ == "__main__":
    test_encoder()



class Decoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.decoder(x)


class SegmentationAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(SegmentationAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

# 训练前，先训练自编码器，之后用训练好的编码器进行分割
encoder = Encoder(in_channels=3, latent_dim=256)
decoder = Decoder(latent_dim=256, out_channels=3)
model = SegmentationAutoencoder(encoder, decoder)
