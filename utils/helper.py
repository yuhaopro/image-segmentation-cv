import torch
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint: str, model, device):
    print("=> Loading checkpoint")
    # original saved file with DataParallel
    checkpoint_info = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_info['state_dict'])    