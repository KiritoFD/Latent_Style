"""
LGT-X (Robust Edition) Model Architecture

Core Design:
- Pure AdaGN-based conditioning (Proven stability)
- Independent style parameters per layer (Fixes Layer Collapse)
- Removed broken Cross-Attention and shared CCMs
- Optimized for RTX 4070 (Channels Last, Memory Efficient)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP projection."""
    def __init__(self, dim=256, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)

class StyleEmbedding(nn.Module):
    """Learnable style embeddings."""
    def __init__(self, num_styles, style_dim=256):
        super().__init__()
        self.embeddings = nn.Embedding(num_styles, style_dim)
        self.register_buffer('avg_embedding', torch.zeros(style_dim))
        self.avg_computed = False
    
    def compute_avg_embedding(self):
        with torch.no_grad():
            self.avg_embedding.copy_(self.embeddings.weight.mean(dim=0))
        self.avg_computed = True
    
    def forward(self, style_id, use_avg=False):
        if use_avg:
            if not self.avg_computed: self.compute_avg_embedding()
            return self.avg_embedding.unsqueeze(0).expand(style_id.shape[0], -1)
        return self.embeddings(style_id)

class AdaGN(nn.Module):
    """
    Adaptive Group Normalization (The Gold Standard for Style Transfer)
    Scale & Shift feature statistics based on style embedding.
    """
    def __init__(self, channels, style_dim, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, affine=False, eps=1e-6)
        self.style_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(style_dim, channels * 2)
        )
        # Init to identity: scale=1, shift=0
        nn.init.zeros_(self.style_proj[1].weight)
        nn.init.zeros_(self.style_proj[1].bias)
        with torch.no_grad():
            self.style_proj[1].bias[:channels] = 1.0

    def forward(self, x, style_emb):
        # x: [B, C, H, W]
        h = self.norm(x)
        style = self.style_proj(style_emb) # [B, 2C]
        style = style.unsqueeze(-1).unsqueeze(-1)
        scale, shift = style.chunk(2, dim=1)
        return h * scale + shift

class LGTXBlock(nn.Module):
    """
    Robust ResBlock with AdaGN.
    """
    def __init__(self, channels, style_dim, dropout=0.0):
        super().__init__()
        self.norm1 = AdaGN(channels, style_dim)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = AdaGN(channels, style_dim)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, style_emb):
        h = self.act(self.norm1(x, style_emb))
        h = self.conv1(h)
        h = self.act(self.norm2(h, style_emb))
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h

class SelfAttention(nn.Module):
    """Global context mixing (Standard Attention)."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x
        x = self.norm(x).view(B, C, -1).permute(0, 2, 1) # [B, N, C]
        qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        h = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        h = self.proj(h).permute(0, 2, 1).reshape(B, C, H, W)
        return x_in + h

class LGTUNet(nn.Module):
    """
    Clean, robust U-Net architecture.
    """
    def __init__(
        self,
        latent_channels=4,
        base_channels=128,
        style_dim=256,
        time_dim=256,
        num_styles=4,
        num_encoder_blocks=2,
        num_decoder_blocks=3,
        **kwargs # Ignore unused args like ccm_rank
    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(time_dim)
        self.style_embed = StyleEmbedding(num_styles, style_dim)
        
        # Fuse time and style
        self.cond_fusion = nn.Sequential(
            nn.Linear(time_dim + style_dim, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim)
        )
        
        # Encoder
        self.in_conv = nn.Conv2d(latent_channels, base_channels, 3, padding=1)
        
        self.enc1 = nn.ModuleList([LGTXBlock(base_channels, style_dim) for _ in range(num_encoder_blocks)])
        self.down1 = nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1) # 32->16
        
        self.enc2 = nn.ModuleList([LGTXBlock(base_channels*2, style_dim) for _ in range(num_encoder_blocks)])
        self.down2 = nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1) # 16->8
        
        # Bottleneck
        self.mid_block1 = LGTXBlock(base_channels*4, style_dim)
        self.attn = SelfAttention(base_channels*4)
        self.mid_block2 = LGTXBlock(base_channels*4, style_dim)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1) # 8->16
        self.dec1 = nn.ModuleList([LGTXBlock(base_channels*2, style_dim) for _ in range(num_decoder_blocks)])
        
        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1) # 16->32
        self.dec2 = nn.ModuleList([LGTXBlock(base_channels, style_dim) for _ in range(num_decoder_blocks)])
        
        self.out_norm = nn.GroupNorm(32, base_channels)
        self.out_conv = nn.Conv2d(base_channels, latent_channels, 3, padding=1)
        
        # Init output to zero
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, style_id, use_avg_style=False):
        t_emb = self.time_embed(t)
        s_emb = self.style_embed(style_id, use_avg=use_avg_style)
        cond = self.cond_fusion(torch.cat([t_emb, s_emb], dim=-1))
        
        h = self.in_conv(x)
        skips = []
        
        # Encoder
        for blk in self.enc1: h = blk(h, cond)
        skips.append(h)
        h = self.down1(h)
        
        for blk in self.enc2: h = blk(h, cond)
        skips.append(h)
        h = self.down2(h)
        
        # Bottleneck
        h = self.mid_block1(h, cond)
        h = self.attn(h)
        h = self.mid_block2(h, cond)
        
        # Decoder (Simple Additive Skip)
        h = self.up1(h)
        h = h + skips.pop() # Residual add
        for blk in self.dec1: h = blk(h, cond)
        
        h = self.up2(h)
        h = h + skips.pop()
        for blk in self.dec2: h = blk(h, cond)
        
        return self.out_conv(F.silu(self.out_norm(h)))

    def compute_avg_style_embedding(self):
        self.style_embed.compute_avg_embedding()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)