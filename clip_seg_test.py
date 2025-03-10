from clip_seg_model import ClipSegmentation
import utils 
import torch.nn as nn
import torch
BATCH_SIZE = 64
LOAD_MODEL = True
DEVICE_NAME = "cuda"
DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model = ClipSegmentation(out_channels=3).to(device=DEVICE)
    if LOAD_MODEL:
        utils.load_checkpoint(torch.load("CLIP_checkpoint.pth.tar"), model)
    test_loader = utils.get_test_loader(batch_size=BATCH_SIZE)
    metric = utils.MetricStorage()
    loss_fn = nn.CrossEntropyLoss()
    utils.check_accuracy(loader=test_loader, model=model, metric=metric, loss_fn=loss_fn, filename="Test")
    

if __name__ == "__main__":
    main()