import torch
import torch.nn as nn

class BasicCNN(nn.Module):
    """Classic 2-layer CNN"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

class DepthwiseCNN(nn.Module):
    """Depthwise separable convolution CNN"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, groups=1), nn.ReLU(),
            nn.Conv2d(1, 16, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.ReLU(),
            nn.Conv2d(16, 32, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

class EfficientCNN(nn.Module):
    """EfficientNet-style compound scaling"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.ReLU(),
            nn.Conv2d(32, 64, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.ReLU(),
            nn.Conv2d(64, 96, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 96, 3, padding=1, groups=96), nn.ReLU(),
            nn.Conv2d(96, 128, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)