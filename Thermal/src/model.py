"""
LGT-X (Robust Edition) Model Architecture

Core Design:
- Pure AdaGN-based conditioning (Proven stability)
- Independent style parameters per layer (Fixes Layer Collapse)
- Removed broken Cross-Attention and shared CCMs
- Optimized for RTX 4070 (Channels Last, Memory Efficient)
"""

import torch
import torch.utils.checkpoint as ckpt
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

class LGTResNet(nn.Module):
    """
    Lightweight conditional ResNet with a tiny multi-scale branch in latent space.
    Outputs velocity field v(x,t,style), same shape as x.
    """

    def __init__(
        self,
        latent_channels=4,
        base_channels=64,
        style_dim=256,
        time_dim=256,
        num_styles=4,
        num_blocks=10,
        num_blocks_pre=None,
        num_blocks_low=None,
        num_blocks_post=None,
        dropout=0.0,
        use_attn=False,
        attn_heads=4,
        v_max=2.0,
        use_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.v_max = float(v_max)
        self.time_embed = TimestepEmbedding(time_dim)
        self.style_embed = StyleEmbedding(num_styles, style_dim)

        self.cond_fusion = nn.Sequential(
            nn.Linear(time_dim + style_dim, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim),
        )

        # Split total blocks across pre/low/post stages if not explicitly provided.
        if num_blocks_pre is None or num_blocks_low is None or num_blocks_post is None:
            default_pre = max(2, int(num_blocks * 0.4))
            default_low = max(1, int(num_blocks * 0.2))
            default_post = max(2, num_blocks - default_pre - default_low)
            num_blocks_pre = default_pre if num_blocks_pre is None else num_blocks_pre
            num_blocks_low = default_low if num_blocks_low is None else num_blocks_low
            num_blocks_post = default_post if num_blocks_post is None else num_blocks_post

        self.in_conv = nn.Conv2d(latent_channels, base_channels, 3, padding=1)
        self.blocks_pre = nn.ModuleList(
            [LGTXBlock(base_channels, style_dim, dropout=dropout) for _ in range(num_blocks_pre)]
        )

        # Tiny-U branch: one downsample stage + a few low-res blocks + upsample back.
        self.downsample = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.blocks_low = nn.ModuleList(
            [LGTXBlock(base_channels, style_dim, dropout=dropout) for _ in range(num_blocks_low)]
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )

        self.blocks_post = nn.ModuleList(
            [LGTXBlock(base_channels, style_dim, dropout=dropout) for _ in range(num_blocks_post)]
        )

        self.use_attn = bool(use_attn)
        self.attn = SelfAttention(base_channels, num_heads=attn_heads) if self.use_attn else None

        out_groups = 32
        while out_groups > 1 and (base_channels % out_groups != 0):
            out_groups //= 2
        self.out_norm = nn.GroupNorm(out_groups, base_channels)
        self.out_conv = nn.Conv2d(base_channels, latent_channels, 3, padding=1)
        self.out_affine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(style_dim, latent_channels * 2),
        )

        # Init output to zero so initial velocity is near zero (stable ODE start)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        nn.init.zeros_(self.out_affine[1].weight)
        nn.init.zeros_(self.out_affine[1].bias)
        with torch.no_grad():
            self.out_affine[1].bias[:latent_channels] = 1.0

    def forward(self, x, t, style_id, use_avg_style=False):
        def _vram(tag: str):
            if getattr(self, "_vram_debug", False):
                fn = getattr(self, "_vram_debug_fn", None)
                if fn is not None:
                    fn(tag)

        def _ckpt(fn, *args):
            if self.use_checkpointing and self.training:
                return ckpt.checkpoint(fn, *args, use_reentrant=False)
            return fn(*args)

        t_emb = self.time_embed(t)
        s_emb = self.style_embed(style_id, use_avg=use_avg_style)
        cond = self.cond_fusion(torch.cat([t_emb, s_emb], dim=-1))

        h = self.in_conv(x)
        _vram("in_conv")

        for blk in self.blocks_pre:
            h = _ckpt(lambda _h, _c, _blk=blk: _blk(_h, _c), h, cond)
        _vram("blocks_pre")

        h_main = h
        h_low = self.downsample(h_main)
        _vram("downsample")
        for blk in self.blocks_low:
            h_low = _ckpt(lambda _h, _c, _blk=blk: _blk(_h, _c), h_low, cond)
        _vram("blocks_low")
        h_low = self.upsample(h_low)
        if h_low.shape[-2:] != h_main.shape[-2:]:
            h_low = F.interpolate(h_low, size=h_main.shape[-2:], mode="bilinear", align_corners=False)
        h = h_main + h_low
        _vram("upsample_fuse")

        if self.use_attn:
            h = _ckpt(lambda _h, _blk=self.attn: _blk(_h), h)
            _vram("attn")

        for blk in self.blocks_post:
            h = _ckpt(lambda _h, _c, _blk=blk: _blk(_h, _c), h, cond)
        _vram("blocks_post")

        raw = self.out_conv(F.silu(self.out_norm(h)))
        style_affine = self.out_affine(cond).unsqueeze(-1).unsqueeze(-1)
        out_scale, out_shift = style_affine.chunk(2, dim=1)
        raw = raw * out_scale + out_shift

        if self.v_max > 0:
            out = self.v_max * torch.tanh(raw / self.v_max)
        else:
            out = raw
        _vram("out")
        return out

    def compute_avg_style_embedding(self):
        self.style_embed.compute_avg_embedding()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
