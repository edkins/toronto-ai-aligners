import torch
from torch import nn

class Layer(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.lin = nn.Linear(count, count)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.lin(x))

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Flatten(),
            Layer(28*28),
            Layer(28*28),
            Layer(28*28),
            Layer(28*28),
            Layer(28*28),
            nn.Linear(28*28, 10)
        )

    def forward(self, x):
        return self.stack(x.to(torch.float32))

    def get_activations(self, x, layernum):
        return self.stack[:layernum](x.to(torch.float32))
