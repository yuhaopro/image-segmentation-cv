import utils
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import random
from autoencoder.model import Autoencoder
from autoencoder.model import AutoencoderWithSegmentationHead

random.seed(42)
BATCH_SIZE = 64
PIN_MEMORY = True
NUM_WORKERS = 4
LEARNING_RATE = 1e-5
LOAD_MODEL = True
NUM_EPOCHS = 50
DEVICE_NAME = "cuda"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_autoencoder(loader, autoencoder, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    epoch_loss = 0

    for batch_idx, (images, _) in enumerate(loop):
        images = images.to(device=DEVICE, dtype=torch.float)

        with torch.autocast(device_type="cuda"):
            decoded = autoencoder(images)
            
            if decoded.shape != images.shape:
                raise ValueError(f"Shape mismatch: decoded {decoded.shape} vs images {images.shape}")

            loss = loss_fn(decoded, images)  # Reconstruction loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Loss became NaN or Inf at batch {batch_idx}")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def train_segmentation_head(loader, segmentation_model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, leave=False)
    epoch_loss = 0

    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device=DEVICE, dtype=torch.float)
        masks = masks.to(device=DEVICE, dtype=torch.long)

        with torch.autocast(device_type="cuda"):
            segmentation_output = segmentation_model(images)
            
            if isinstance(segmentation_output, tuple):
                segmentation_output = segmentation_output[0]  # Extract the first element as output
            
                


        # check the shapes of masks meet the expectation of CrossEntropyLoss
        if masks.dim() != 3:
            raise ValueError(f"Expected masks to have 3 dimensions (B, H, W), but got {masks.shape}")

        with torch.autocast(device_type="cuda"):
            segmentation_output = segmentation_model(images)
            
            # if the output is a tuple, get the second element
            if isinstance(segmentation_output, tuple):
                segmentation_output = segmentation_output[1]

            # check if the output shape is correct
            if segmentation_output.shape[0] != masks.shape[0]:
                raise ValueError(f"Output batch size {segmentation_output.shape[0]} does not match mask batch size {masks.shape[0]}")

            loss = loss_fn(segmentation_output, masks)

            # check if loss value is reasonable
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

    # check if the data is successfully loaded
    train_loader, val_loader = utils.get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    if train_loader is None or val_loader is None:
        raise RuntimeError("Error: Failed to load training or validation data.")

    # Initialize Autoencoder
    autoencoder = Autoencoder().to(DEVICE)
    autoencoder_loss_fn = nn.MSELoss()
    autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    scaler = torch.GradScaler()

    # train autoencoder
    for epoch in range(NUM_EPOCHS):
        epoch_loss = train_autoencoder(train_loader, autoencoder, autoencoder_optimizer, autoencoder_loss_fn, scaler)
        print(f"Autoencoder Epoch {epoch+1}, Loss: {epoch_loss}")

        utils.save_checkpoint({"state_dict": autoencoder.state_dict()},
                               filename=f"autoencoder_checkpoint_{epoch}.pth.tar")

    # check if the encoder is successfully extracted
    if not hasattr(autoencoder, "encoder"):
        raise RuntimeError("Error: Autoencoder does not have an encoder attribute.")

    # initialize
    segmentation_model = AutoencoderWithSegmentationHead(autoencoder.encoder, autoencoder.decoder).to(DEVICE)
    segmentation_loss_fn = nn.CrossEntropyLoss()
    segmentation_optimizer = optim.Adam(segmentation_model.parameters(), lr=1e-3)

    # train segmentation head
    for epoch in range(50):
        epoch_loss = train_segmentation_head(train_loader, segmentation_model, segmentation_optimizer, segmentation_loss_fn, scaler)
        print(f"Segmentation Epoch {epoch+1}, Loss: {epoch_loss}")

        utils.save_checkpoint({"state_dict": segmentation_model.state_dict()},
                               filename=f"segmentation_model_checkpoint_{epoch}.pth.tar")

    print("Training finished...")

if __name__ == "__main__":
    train()
