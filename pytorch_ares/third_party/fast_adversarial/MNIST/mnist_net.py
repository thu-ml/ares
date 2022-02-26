import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mnist_net():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model
