"""
LGT-X (Robust Edition) Model Architecture

Core Design:
- Pure AdaGN-based conditioning (Proven stability)
- Independent style parameters per layer (Fixes Layer Collapse)
- Removed broken Cross-Attention and shared CCMs
- Optimized for RTX 4070 (Channels Last, Memory Efficient)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP projection."""
    def __init__(self, dim=64, max_period=10000):
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError(f"TimestepEmbedding dim must be an even integer >= 2, got {dim}")
        self.dim = dim
        self.max_period = max_period
        hidden_dim = dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, t):
        t = t.float().clamp(0.0, 1.0)
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

class LGTUNetLite(nn.Module):
    """
    Lightweight U-Net in latent space.
    - AdaGN conditioning on every block
    - No attention
    - Upsampling via interpolate + conv for cleaner artifacts
    - Outputs velocity field v(x, t, style)
    """

    def __init__(
        self,
        latent_channels=4,
        base_channels=64,
        style_dim=256,
        time_dim=64,
        num_styles=4,
        num_encoder_blocks=1,
        num_decoder_blocks=1,
        dropout=0.0,
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

        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4

        self.in_conv = nn.Conv2d(latent_channels, ch1, 3, padding=1)

        self.enc1 = nn.ModuleList([LGTXBlock(ch1, style_dim, dropout=dropout) for _ in range(num_encoder_blocks)])
        self.down1 = nn.Conv2d(ch1, ch2, 3, stride=2, padding=1)

        self.enc2 = nn.ModuleList([LGTXBlock(ch2, style_dim, dropout=dropout) for _ in range(num_encoder_blocks)])
        self.down2 = nn.Conv2d(ch2, ch3, 3, stride=2, padding=1)

        self.mid1 = LGTXBlock(ch3, style_dim, dropout=dropout)
        self.mid2 = LGTXBlock(ch3, style_dim, dropout=dropout)

        self.up1_conv = nn.Conv2d(ch3, ch2, 3, padding=1)
        self.dec1 = nn.ModuleList([LGTXBlock(ch2, style_dim, dropout=dropout) for _ in range(num_decoder_blocks)])

        self.up2_conv = nn.Conv2d(ch2, ch1, 3, padding=1)
        self.dec2 = nn.ModuleList([LGTXBlock(ch1, style_dim, dropout=dropout) for _ in range(num_decoder_blocks)])

        out_groups = 32
        while out_groups > 1 and (ch1 % out_groups != 0):
            out_groups //= 2
        self.out_norm = nn.GroupNorm(out_groups, ch1, eps=1e-6)
        self.out_conv = nn.Conv2d(ch1, latent_channels, 3, padding=1)

        # Start near zero-velocity for ODE stability.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x,
        t,
        style_id,
        use_avg_style=False,
        return_features: bool = False,
        feature_levels=("early", "mid", "late"),
    ):
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
        requested_levels = set(feature_levels) if return_features else set()
        features = {} if return_features else None

        h = self.in_conv(x)
        _vram("in_conv")
        skips = []

        for blk in self.enc1:
            h = _ckpt(lambda _h, _c, _blk=blk: _blk(_h, _c), h, cond)
        if "early" in requested_levels:
            features["early"] = h
        skips.append(h)
        h = self.down1(h)
        _vram("down1")

        for blk in self.enc2:
            h = _ckpt(lambda _h, _c, _blk=blk: _blk(_h, _c), h, cond)
        skips.append(h)
        h = self.down2(h)
        _vram("down2")

        h = _ckpt(lambda _h, _c: self.mid1(_h, _c), h, cond)
        h = _ckpt(lambda _h, _c: self.mid2(_h, _c), h, cond)
        if "mid" in requested_levels:
            features["mid"] = h
        _vram("mid")

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up1_conv(h)
        skip = skips.pop()
        if h.shape[-2:] != skip.shape[-2:]:
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
        h = h + skip
        for blk in self.dec1:
            h = _ckpt(lambda _h, _c, _blk=blk: _blk(_h, _c), h, cond)
        _vram("up1")

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up2_conv(h)
        skip = skips.pop()
        if h.shape[-2:] != skip.shape[-2:]:
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
        h = h + skip
        for blk in self.dec2:
            h = _ckpt(lambda _h, _c, _blk=blk: _blk(_h, _c), h, cond)
        if "late" in requested_levels:
            features["late"] = h
        _vram("up2")

        raw = self.out_conv(F.silu(self.out_norm(h)))
        if self.v_max > 0:
            out = self.v_max * torch.tanh(raw / self.v_max)
        else:
            out = raw
        _vram("out")
        if return_features:
            return out, features
        return out

    def compute_avg_style_embedding(self):
        self.style_embed.compute_avg_embedding()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
