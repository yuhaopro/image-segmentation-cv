import utils
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import random
from autoencoder import Autoencoder

random.seed(42)
BATCH_SIZE = 16
PIN_MEMORY = True
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
LOAD_MODEL = True
NUM_EPOCHS = 40
DEVICE_NAME = "cuda"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_autoencoder(loader, autoencoder, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    epoch_loss = 0

    for batch_idx, (images, _) in enumerate(loop):
        images = images.to(device=DEVICE, dtype=torch.float)

        with torch.autocast(device_type="cuda"):
            decoded = autoencoder(images)
            
            # **检查：确保输出形状匹配**
            if decoded.shape != images.shape:
                raise ValueError(f"Shape mismatch: decoded {decoded.shape} vs images {images.shape}")

            loss = loss_fn(decoded, images)  # 重建误差
            
            # **检查：loss 值是否合理**
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Loss became NaN or Inf at batch {batch_idx}")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def train():
    print("start")

    # **检查：确保数据加载成功**
    train_loader, val_loader = utils.get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    if train_loader is None or val_loader is None:
        raise RuntimeError("Error: Failed to load training or validation data.")

    # 初始化 Autoencoder
    autoencoder = Autoencoder().to(DEVICE)
    autoencoder_loss_fn = nn.MSELoss()
    autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    scaler = torch.GradScaler()

    # **训练自编码器**
    for epoch in range(NUM_EPOCHS):
        epoch_loss = train_autoencoder(train_loader, autoencoder, autoencoder_optimizer, autoencoder_loss_fn, scaler)
        print(f"Autoencoder Epoch {epoch+1}, Loss: {epoch_loss}")

        utils.save_checkpoint({"state_dict": autoencoder.state_dict()},
                               filename=f"autoencoder_checkpoint_{epoch}.pth.tar")

    print("Training finished...")

if __name__ == "__main__":
    train()
