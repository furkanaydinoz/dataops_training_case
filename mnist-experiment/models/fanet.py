"""
FANET - Feature Adaptive Network with Enhanced Transformers
A hybrid CNN + Transformer architecture for MNIST classification.

Key Innovation:
- MultiScaleCNNStem: Hierarchical feature extraction (unlike ViT's linear patch projection)
- ConvolutionalMHSA: Depthwise convolutions in QKV projections for local+global attention
- Hybrid design combines CNN's locality with Transformer's global context
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, kernel_size // 2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else \
                        nn.Conv2d(in_channels, out_channels, 1, stride)

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class MultiScaleCNNStem(nn.Module):
    """
    Hierarchical CNN feature extractor.
    Unlike ViT's linear patch projection, this extracts multi-scale features.
    Output: (B, 128, 4, 4) for 28x28 MNIST input
    """
    def __init__(self, in_channels=1, channels=[32, 64, 128]):
        super().__init__()
        stages = []
        for i, out_ch in enumerate(channels):
            stride = 2
            stages.append(DepthwiseSeparableConv(
                channels[i - 1] if i > 0 else in_channels,
                out_ch, stride=stride
            ))
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        return self.stages(x)


class ConvolutionalMHSA(nn.Module):
    """
    Multi-Head Attention with convolutional QKV projections.
    Key difference from standard MHSA: uses depthwise conv instead of linear projections.
    This adds local inductive bias to the attention mechanism.
    """
    def __init__(self, d_model, num_heads, kernel_size=3):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Use depthwise conv for local inductive bias in QKV projections
        self.q_conv = nn.Conv2d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.k_conv = nn.Conv2d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.v_conv = nn.Conv2d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        # Convolutional QKV projections
        q = self.q_conv(x).reshape(B, self.num_heads, self.d_k, H * W)
        k = self.k_conv(x).reshape(B, self.num_heads, self.d_k, H * W)
        v = self.v_conv(x).reshape(B, self.num_heads, self.d_k, H * W)
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_k)
        attn = torch.softmax(q.transpose(-2, -1) @ k * scale, dim=-1)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        return self.out_proj(out.reshape(B, C, H, W))


class HybridConvTransformerBlock(nn.Module):
    """
    Transformer block with convolutional attention.
    Combines local feature extraction (CNN) with global context (Attention).
    """
    def __init__(self, d_model, num_heads, mlp_ratio=4, kernel_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ConvolutionalMHSA(d_model, num_heads, kernel_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to sequence for LayerNorm
        x_flat = x.reshape(B, C, H * W).transpose(1, 2)  # (B, N, C)
        # Attention with residual
        attn_out = self.attn(self.norm1(x_flat).transpose(1, 2).reshape(B, C, H, W))
        x_flat = x_flat + attn_out.reshape(B, H * W, C).transpose(1, 2)
        # FFN with residual
        x_flat = x_flat + self.ff(self.norm2(x_flat))
        return x_flat.transpose(1, 2).reshape(B, C, H, W)


class FANET(nn.Module):
    """
    Feature Adaptive Network with Enhanced Transformers.

    Architecture:
    1. MultiScaleCNNStem - Extracts hierarchical features (unlike ViT's linear projection)
    2. HybridConvTransformerBlocks - Combines local CNN attention with global context
    3. Global Average Pooling + Classification head

    Comparison with ViT:
    - ViT: Linear patch projection → Pure global attention
    - FANET: Hierarchical CNN features → Local+global attention via conv in QKV
    """
    def __init__(self, in_channels=1, cnn_channels=[32, 64, 128], embed_dim=128, num_heads=4, depth=3, num_classes=10):
        super().__init__()
        self.stem = MultiScaleCNNStem(in_channels, cnn_channels)
        self.pos_embed = nn.Parameter(torch.randn(1, 128, 4, 4) * 0.02)
        self.blocks = nn.ModuleList([
            HybridConvTransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)  # (B, 128, 4, 4)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[2, 3])  # Global average pool over spatial dims
        return self.head(self.norm(x))