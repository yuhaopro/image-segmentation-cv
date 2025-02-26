from transformers import TrainingArguments, Trainer
import pickle as pkl
from UNET_model import UNET
from tqdm import tqdm
import torch 

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2

with open("processed_dataset/train.pkl", "rb") as f:
    train_dataset = pkl.load(f)

with open("processed_dataset/val.pkl", "rb") as f:
    val_dataset = pkl.load(f)

with open("processed_dataset/test.pkl", "rb") as f:
    test_dataset = pkl.load(f)


def train(loader, model, optimizer, loss_fn, scaler):
    print(f"Starting training...")
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    pass

if __name__ == "__main__":
    main()