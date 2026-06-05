import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """3-layer baseline MLP"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

class WideMLP(nn.Module):
    """Wide shallow MLP - depth yerine width"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)

class DeepMLP(nn.Module):
    """Deep narrow MLP - width yerine depth"""
    def __init__(self):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(784, 64), nn.ReLU()]
        for _ in range(6):
            layers.extend([nn.Linear(64, 64), nn.ReLU()])
        layers.append(nn.Linear(64, 10))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)