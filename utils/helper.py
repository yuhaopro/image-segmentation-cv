import torch

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, device):
    print("=> Loading checkpoint")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    