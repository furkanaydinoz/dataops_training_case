"""
FANET-MicroGPT - Feature Adaptive Network with Enhanced Transformers for Language Modeling
A hybrid CNN + Transformer architecture adapted for sequence modeling.

Key Innovation:
- Conv1DStem: 1D convolution stem for sequence processing
- ConvolutionalMHSA1D: 1D depthwise convolutions in QKV for local attention
- Maintains transformer architecture but with enhanced attention mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Conv1DStem(nn.Module):
    """
    1D Convolution stem for sequence modeling.
    Processes embedding sequence through depthwise separable convolutions.
    Adds local inductive bias to the transformer.
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv_stages = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.embed(x)  # (B, T, E)
        return self.conv_stages(x.transpose(1, 2)).transpose(1, 2)  # (B, T, E)


class ConvolutionalMHSA1D(nn.Module):
    """
    Multi-Head Attention with 1D convolutional QKV projections.
    Key difference from standard MHSA: uses depthwise conv instead of linear projections.
    This adds local receptive field to the attention mechanism.
    """
    def __init__(self, d_model, num_heads, kernel_size=3):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Use depthwise conv for local inductive bias in QKV projections
        self.q_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.k_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.v_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        # Convolutional QKV projections
        q = self.q_conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, C)
        k = self.k_conv(x.transpose(1, 2)).transpose(1, 2)
        v = self.v_conv(x.transpose(1, 2)).transpose(1, 2)
        # Reshape for multi-head attention
        q = q.reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = k.reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_k)
        scores = (q @ k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Standard feed-forward network with GELU activation"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FANETTransformerBlock(nn.Module):
    """
    Transformer block with convolutional attention.
    Pre-norm architecture with residual connections.
    """
    def __init__(self, d_model, num_heads, dropout=0.1, kernel_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ConvolutionalMHSA1D(d_model, num_heads, kernel_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.ff(self.norm2(x))
        return x


class FANETMicroGPT(nn.Module):
    """
    FANET-Enhanced MicroGPT for language modeling.

    Architecture:
    1. Conv1DStem - Local feature extraction from token embeddings
    2. FANETTransformerBlocks - Hybrid conv-attention transformer
    3. LayerNorm + Language modeling head

    Comparison with original MicroGPT:
    - MicroGPT: Linear token embeddings → Pure global attention
    - FANET-MicroGPT: Conv-stem processing → Local+global attention
    """
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=4, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        # Use Conv1DStem instead of standard embedding
        self.stem = Conv1DStem(vocab_size, d_model)
        self.position_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            FANETTransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
        # Conv-stem processing
        x = self.stem(x)  # (B, T, d_model)
        # Position embedding
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.position_embed(pos)
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_new_tokens=100, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            x_crop = x[:, -self.max_seq_len:]
            logits, _ = self.forward(x_crop)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
        return x