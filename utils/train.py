import torch
import torchvision
import torch.nn.functional as F
import os

class EarlyStopping:
    def __init__(self, min_delta=0.001, patience=2):
        self.min_delta = min_delta
        self.patience = patience
        self.best = float("inf")
        self.wait = 0
        self.done = False

    def step(self, current):
        self.wait += 1

        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        elif self.wait >= self.patience:
            self.done = True

        return self.done

def log_training(epoch, loss, best, wait):
        print(
        f"{epoch=:03}"
        f"{loss=:.02f}"
        f"best={best:.02f}"
        f"wait={wait}"
    )
     

