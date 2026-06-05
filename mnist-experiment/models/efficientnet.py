import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            Swish(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.net(x).view(x.shape[0], -1, 1, 1)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=1, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        hidden_dim = int(in_channels * expand_ratio)
        layers = []
        if expand_ratio != 1:
            layers.extend([nn.Conv2d(in_channels, hidden_dim, 1), nn.BatchNorm2d(hidden_dim), Swish()])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim), Swish(),
            SEBlock(hidden_dim, reduction=int(1/se_ratio)) if se_ratio > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, out_channels, 1), nn.BatchNorm2d(out_channels)
        ])
        self.net = nn.Sequential(*layers)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride), nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return torch.relu(self.net(x) + self.shortcut(x))

class EfficientNetStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), Swish(),
            MBConvBlock(32, 32, expand_ratio=1),
            MBConvBlock(32, 48, stride=2),
            MBConvBlock(48, 64, expand_ratio=2),
            MBConvBlock(64, 96, stride=2),
            MBConvBlock(96, 96),
            MBConvBlock(96, 128, expand_ratio=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.stem(x)