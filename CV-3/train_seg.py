import utils
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import random
from autoencoder import AutoencoderWithSegmentationHead
from autoencoder import Autoencoder
random.seed(42)
BATCH_SIZE = 64
PIN_MEMORY = True
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
LOAD_MODEL = True
NUM_EPOCHS = 40
DEVICE_NAME = "cuda"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_segmentation_head(loader, segmentation_model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, leave=False)
    epoch_loss = 0

    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device=DEVICE, dtype=torch.float)
        masks = masks.to(device=DEVICE, dtype=torch.long)

        with torch.autocast(device_type="cuda"):
            segmentation_output = segmentation_model(images)
            
            
            if isinstance(segmentation_output, tuple):
                segmentation_output = segmentation_output[0]  

        
        if masks.dim() != 3:
            raise ValueError(f"Expected masks to have 3 dimensions (B, H, W), but got {masks.shape}")

        with torch.autocast(device_type="cuda"):
            
            segmentation_output = segmentation_model(images)

            
            if segmentation_output.shape[0] != masks.shape[0]:
                raise ValueError(f"Output batch size {segmentation_output.shape[0]} does not match mask batch size {masks.shape[0]}")

            loss = loss_fn(segmentation_output, masks)

            
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Loss became NaN or Inf at batch {batch_idx}")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def train_segmentation_model(autoencoder):
    print("Start training Segmentation Head...")

    train_loader, val_loader = utils.get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    if train_loader is None or val_loader is None:
        raise RuntimeError("Error: Failed to load training or validation data.")

    
    segmentation_model = AutoencoderWithSegmentationHead(autoencoder.encoder).to(DEVICE)
    segmentation_loss_fn = nn.CrossEntropyLoss()
    segmentation_optimizer = optim.Adam(segmentation_model.parameters(), lr=LEARNING_RATE)
    scaler = torch.GradScaler()

    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = train_segmentation_head(train_loader, segmentation_model, segmentation_optimizer, segmentation_loss_fn, scaler)
        print(f"Segmentation Epoch {epoch+1}, Loss: {epoch_loss}")

        utils.save_checkpoint({"state_dict": segmentation_model.state_dict()},
                               filename=f"segmentation_model_checkpoint_{epoch}.pth.tar")

    print("Segmentation training finished...")

if __name__ == "__main__":
    
    autoencoder = Autoencoder().to(DEVICE)
    checkpoint = torch.load('autoencoder_checkpoint_18.pth.tar', map_location=DEVICE)  # 使用最后一个保存的 Autoencoder 检查点
    autoencoder.load_state_dict(checkpoint['state_dict'])

    train_segmentation_model(autoencoder)
