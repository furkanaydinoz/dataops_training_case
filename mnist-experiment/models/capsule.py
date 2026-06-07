import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_capsules=10, dim=16, kernel_size=9, stride=2):
        super().__init__()
        self.num_capsules = num_capsules
        self.dim = dim
        self.conv = nn.Conv2d(in_channels, out_channels * dim, kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        batch = x.shape[0]
        x = self.conv(x)
        x = x.view(batch, self.num_capsules, self.dim, -1).permute(0, 1, 3, 2)
        x = x.reshape(batch, self.num_capsules, -1)
        x = F.normalize(x, dim=-1)
        return x

class CapsuleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # conv1: 28x28 -> 20x20
        self.conv1 = nn.Conv2d(1, 256, 9)
        # caps1: produces 512 channels (64 * 8) for caps2 which expects 512
        self.caps1 = CapsuleLayer(256, 64, num_capsules=8, dim=8, kernel_size=5, stride=2)
        # caps2: input=512 channels, output=10 capsules * 16 dim
        self.caps2 = CapsuleLayer(512, 10, num_capsules=10, dim=16, kernel_size=3, stride=1)
        self.decoder = nn.Sequential(
            nn.Linear(16*10, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 784), nn.Sigmoid()
        )
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.caps1(x)
        x = self.caps2(x)
        x = x.norm(dim=-1)
        return x