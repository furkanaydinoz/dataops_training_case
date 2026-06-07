import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryCaps(nn.Module):
    """Primary Capsule Layer - converts features to capsule representation"""
    def __init__(self, in_channels, out_capsules, capsule_dim, kernel_size=5, stride=2):
        super().__init__()
        self.out_capsules = out_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, out_capsules * capsule_dim,
                              kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        batch = x.size(0)
        out = self.conv(x)  # (B, out_capsules * capsule_dim, H, W)
        H, W = out.shape[2], out.shape[3]
        out = out.view(batch, self.out_capsules, self.capsule_dim, H, W)  # (B, out_caps, dim, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # (B, out_caps, H, W, dim)
        out = out.view(batch, self.out_capsules, H * W, self.capsule_dim)  # (B, out_caps, H*W, dim)
        out = F.normalize(out, dim=-1)
        return out


class DigitCaps(nn.Module):
    """Digit Capsule Layer - routes between capsules"""
    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, num_routing=3):
        super().__init__()
        self.in_capsules = in_capsules
        self.in_dim = in_dim
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.num_routing = num_routing
        self.W = nn.Parameter(torch.randn(in_capsules, in_dim, out_capsules, out_dim))
        nn.init.normal_(self.W, 0, 0.5)

    def forward(self, u):
        batch = u.size(0)
        u = u.unsqueeze(2)  # (B, in_caps, 1, in_dim)
        u = u.unsqueeze(3)  # (B, in_caps, 1, 1, in_dim)
        u_hat = torch.matmul(u, self.W)  # (B, in_caps, out_caps, 1, out_dim)
        u_hat = u_hat.squeeze(3)  # (B, in_caps, out_caps, out_dim)

        # Dynamic routing
        b = torch.zeros(batch, self.in_capsules, self.out_capsules, 1, device=u.device)
        for _ in range(self.num_routing):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1)  # (B, out_caps, out_dim)
            v = F.normalize(s, dim=-1)
            if _ < self.num_routing - 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1, keepdim=True)

        return v.squeeze(-1)  # (B, out_caps, out_dim)


class CapsuleNetwork(nn.Module):
    """Capsule Network for MNIST classification

    Architecture:
    - Feature extraction via conv + batch norm
    - PrimaryCaps: 8 capsule types, each with dim=8, at 12x12 spatial positions
    - DigitCaps: Dynamic routing from 1152 input capsules (8 types × 144 positions) to 10 output capsules
    - Decoder: Reconstruction head for auxiliary loss
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Feature extraction: 1 channel -> 64 channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(64)

        # Primary capsules: produces 8 capsule types at 12x12 spatial positions
        # Input: (B, 64, 28, 28) -> Output: (B, 8, 144, 8)
        # After stride=2, kernel=5, padding=0: (28-5)/2+1 = 12 spatial
        self.primary_caps = PrimaryCaps(64, out_capsules=8, capsule_dim=8, kernel_size=5, stride=2)

        # DigitCaps: routes from 1152 input capsules (8 types × 144 positions) to 10 outputs
        # Each input capsule has dim=8
        self.digit_caps = DigitCaps(in_capsules=1152, in_dim=8, out_capsules=num_classes,
                                     out_dim=16, num_routing=3)

        # Decoder for reconstruction loss
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch = x.size(0)

        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, 28, 28)

        # Primary capsules: (B, 64, 28, 28) -> (B, 8, 144, 8)
        x = self.primary_caps(x)

        # Reshape for digit caps: (B, 8, 144, 8) -> (B, 1152, 8)
        x = x.reshape(batch, 8, -1, 8)
        x = x.reshape(batch, -1, 8)  # (B, 1152, 8)

        # Digit capsules: (B, 1152, 8) -> (B, 10, 16)
        x = self.digit_caps(x)

        # Capsule lengths as class scores
        classes = (x ** 2).sum(dim=-1).sqrt()  # (B, 10)

        # Reconstruction during training
        if self.training:
            _, max_idx = classes.max(dim=1)
            masked = torch.zeros_like(x)
            for i in range(batch):
                masked[i, max_idx[i], :] = x[i, max_idx[i], :]

            reconstruction = self.decoder(masked.view(batch, -1))
            return classes, reconstruction.view(batch, 1, 28, 28)

        return classes