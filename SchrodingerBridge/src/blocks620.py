"""Spatial Bridge Block 620 — cleaned active path only.

628/629 cleanup: removed dead attn_modes (gated/gated_raw/style_select/sparsemax),
FiLM modulation, style MoE, learnable shortcut, skip_coarse,
top-k truncation, style_bias. Kept RMSNorm (E4+ uses it) and softmax/relu2 modes.
630 Phase 6 (DINO 退役): content_dino query path removed entirely.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectral620 import dwt2_haar, idwt2_haar


def _make_norm(dim: int) -> nn.Module:
    """Normalization layer (RMSNorm removed — GroupNorm hardcoded)."""
    return nn.GroupNorm(1, dim, affine=False)


class SpatialBridgeBlock620(nn.Module):
    """AdaLN(time) → Self-Attention → Cross-Attention(style) → FFN.

    Active path (clean_base_v2): relu2 attention, tanh_gate, single k/v proj,
    no FiLM, no MoE, no skip_coarse.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        style_gate_init: float = 0.5,
        style_shortcut_alpha: Any = 1.0,
        layer_idx: int = 0,
        num_layers: int = 4,
        attn_temperature: float = 1.0,
        gate_warmup_steps: int = 0,
        dwt_route: bool = False,
        dwt_ll_route_alpha: float = 0.0,
        dwt_route_train_prob: float = 0.0,
        ll_adaln_zero: bool = False,
        ll_tone_bias: bool = False,
        attn_mode: str = "relu2",
    ) -> None:
        super().__init__()

        self.layer_idx = int(layer_idx)
        self.num_layers = int(num_layers)
        self.gate_warmup_steps = max(0, int(gate_warmup_steps))
        self._current_step = 0
        self.max_entropy_queries = 256
        self.cross_attn_entropy = torch.tensor(0.0)
        self.cross_attn_guidance: torch.Tensor | None = None
        # 630 Phase 4J.1: DWT-Routed Cross-Attention (方案 B)
        self.dwt_route = bool(dwt_route)
        # 630 Remote T2: Soft LL Route — LL 以 alpha 残差注入 style
        self.dwt_ll_route_alpha = float(dwt_ll_route_alpha)
        # 630 Local T10: Stochastic DWT Route — 训练时以概率p使用DWT route, 推理时始终使用
        self.dwt_route_train_prob = float(dwt_route_train_prob)
        # 630 Phase 72 方案 C: Global AdaLN-Zero on LL
        # 独立 global_tone_embedding 通过 AdaLN-Zero 调制 LL 的 mean/std
        # γ/β 零初始化 (训练初期恒等), 逐渐学习色调调制, 不破坏边缘结构
        self.ll_adaln_zero = bool(ll_adaln_zero)
        # 630 Phase 72 方案 D: Direct Tone Bias Injection (无 GroupNorm, 强制注入)
        self.ll_tone_bias = bool(ll_tone_bias)
        self.shortcut_alpha = float(style_shortcut_alpha) if not isinstance(style_shortcut_alpha, (list, tuple)) else (
            float(style_shortcut_alpha[self.layer_idx]) if self.layer_idx < len(style_shortcut_alpha) else 1.0
        )
        self.dim = int(dim)
        self.num_heads = max(1, min(int(num_heads), self.dim))
        while self.dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.dim // self.num_heads
        # 630 Phase 72 方案 C: AdaLN-Zero modules (必须在 self.dim 赋值之后初始化)
        if self.ll_adaln_zero:
            self.adaln_norm = nn.GroupNorm(min(self.dim, 8), self.dim)
            self.adaln_proj = nn.Linear(self.dim, self.dim * 2)
            nn.init.zeros_(self.adaln_proj.weight)
            nn.init.zeros_(self.adaln_proj.bias)
        # 630 Phase 72 方案 D: Direct Tone Bias modules (无 GroupNorm, scale+shift 直接注入)
        # LL = LL * (1 + α*γ) + α*β, γ/β 从 global_tone 投影, α 可学习标量 init=0.1
        if self.ll_tone_bias:
            self.tone_proj = nn.Linear(self.dim, self.dim * 2)
            nn.init.normal_(self.tone_proj.weight, std=0.01)
            nn.init.zeros_(self.tone_proj.bias)
            self.tone_alpha = nn.Parameter(torch.tensor(0.1))
        # 630 Phase 72 清理: norm_type/group_norm, attn_mode/relu2, gate_mode/tanh_gate 硬编码 (已验证最优)
        self.norm1 = _make_norm(self.dim)
        self.time_adaln = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, self.dim * 3))
        # Self-attention: content Q/K/V
        self.sa_qkv = nn.Linear(self.dim, self.dim * 3)
        self.sa_out = nn.Linear(self.dim, self.dim)
        # Cross-attention: content Q, style K/V (single projection, no MoE)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.norm2 = _make_norm(self.dim)
        self.ffn = nn.Sequential(
            _make_norm(self.dim),
            nn.Conv2d(self.dim, self.dim * 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(self.dim * 4, self.dim, kernel_size=1),
        )
        self.style_gate = nn.Parameter(torch.tensor(float(style_gate_init)))
        self._gate_init = float(style_gate_init)
        self.attn_mode = str(attn_mode).lower().strip()  # default "relu2" (629 D19-D22 最优); "softmax" 用 flash attention 适配像素空间大分辨率
        self.attn_temperature = float(attn_temperature)
        nn.init.zeros_(self.time_adaln[-1].weight)
        nn.init.zeros_(self.time_adaln[-1].bias)
        nn.init.zeros_(self.sa_out.bias)
        nn.init.zeros_(self.out_proj.bias)
        self.last_debug: dict[str, torch.Tensor] = {}

    def set_step(self, step: int) -> None:
        """Update the current training step for gate warmup scheduling."""
        self._current_step = int(step)

    def _effective_gate_value(self) -> torch.Tensor:
        """Compute effective gate value with warmup schedule."""
        raw = torch.tanh(self.style_gate)
        if self.gate_warmup_steps <= 0 or not self.training:
            return raw
        warmup_factor = min(1.0, self._current_step / max(1, self.gate_warmup_steps))
        return raw * warmup_factor

    def _compute_use_dwt(self) -> bool:
        """Decide whether to use DWT-routed cross-attention for this forward call.

        - Inference: always use DWT route when enabled (LL bypass protects content).
        - Training with dwt_route_train_prob > 0: stochastic route (T11 SOTA config),
          so q_proj / style_mem jointly learn DWT coefficients and full style.
        - Training with dwt_route_train_prob == 0: deterministic route when enabled
          (4J.1 original behavior).
        """
        if not self.dwt_route:
            return False
        if not self.training:
            return True
        if self.dwt_route_train_prob > 0.0:
            return torch.rand(1).item() < self.dwt_route_train_prob
        return True

    def forward(
        self,
        x: torch.Tensor,
        *,
        time_emb: torch.Tensor,
        style_tokens: torch.Tensor,
        style_global: torch.Tensor | None = None,
        global_tone: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del style_global  # unused (FiLM removed)
        b, c, h, w = x.shape

        # --- Self-attention with AdaLN(time) ---
        normed1 = self.norm1(x)
        scale, shift, gate_t = self.time_adaln(time_emb).to(dtype=x.dtype).chunk(3, dim=1)
        h_time = normed1 * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        sa_in = h_time.permute(0, 2, 3, 1).reshape(b, h * w, c)
        sa_qkv = self.sa_qkv(sa_in)
        sa_q, sa_k, sa_v = sa_qkv.chunk(3, dim=-1)
        sa_q = sa_q.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        sa_k = sa_k.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        sa_v = sa_v.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        sa_out = F.scaled_dot_product_attention(sa_q, sa_k, sa_v, dropout_p=0.0, is_causal=False)
        sa_out = sa_out.transpose(1, 2).reshape(b, h * w, c)
        sa_out = self.sa_out(sa_out)
        time_gate = torch.sigmoid(gate_t[:, :, None, None]).to(dtype=x.dtype)
        sa_delta = time_gate * sa_out.transpose(1, 2).reshape(b, c, h, w)
        x = x + sa_delta

        # --- Cross-attention (content × style) ---
        # 630 Phase 4J.1: DWT-Routed Cross-Attention (方案 B)
        # 理论: 对特征图做 Haar DWT, LL bypass (保结构), 仅高频(LH/HL/HH) query style_mem
        # 解放 style_mem: 100% 容量表达笔触/色彩, 不再被迫学"维持结构"
        ca_is_dwt = False
        use_dwt = self._compute_use_dwt()
        if use_dwt and h >= 2 and w >= 2 and (h % 2 == 0) and (w % 2 == 0):
            x_f = x.float()
            ll_f, lh_f, hl_f, hh_f = dwt2_haar(x_f)
            hf_h, hf_w = ll_f.shape[-2], ll_f.shape[-1]
            lh_tokens = lh_f.permute(0, 2, 3, 1).reshape(b, hf_h * hf_w, c)
            hl_tokens = hl_f.permute(0, 2, 3, 1).reshape(b, hf_h * hf_w, c)
            hh_tokens = hh_f.permute(0, 2, 3, 1).reshape(b, hf_h * hf_w, c)
            if self.dwt_ll_route_alpha > 0.0:
                # 630 Remote T2: Soft LL Route — LL 也参与 cross-attention query
                ll_tokens = ll_f.permute(0, 2, 3, 1).reshape(b, hf_h * hf_w, c)
                ca_in = torch.cat([ll_tokens, lh_tokens, hl_tokens, hh_tokens], dim=1)
                ca_h, ca_w = 4 * hf_h, hf_w
            else:
                ca_in = torch.cat([lh_tokens, hl_tokens, hh_tokens], dim=1)
                ca_h, ca_w = 3 * hf_h, hf_w
            ca_is_dwt = True
        else:
            ca_in = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            ca_h, ca_w = h, w
        q = self.q_proj(ca_in).view(b, ca_in.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k_tokens = self.k_proj(style_tokens)
        v_tokens = self.v_proj(style_tokens)
        k = k_tokens.view(b, style_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v_tokens.view(b, style_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        attn_entropy, pixel_entropy = self._attention_stats(q, k, h=ca_h, w=ca_w, dtype=x.dtype)

        # --- Attention computation (only relu2 + softmax kept) ---
        scale_attn = 1.0 / math.sqrt(float(self.head_dim))
        temp = max(self.attn_temperature, 1e-4)
        if self.attn_mode == "relu2":
            # ReLU^2 attention: sparse, magnitude-preserving (clean_base_v2 active mode)
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale_attn / temp
            gates = torch.relu(logits) ** 2
            attended = torch.matmul(gates, v)
            with torch.no_grad():
                gate_mean = gates.detach().float().mean()
                gate_std = gates.detach().float().std()
                active_ratio = (gates > 0.0).float().mean()
                actual_attn_entropy = -torch.log(active_ratio.clamp_min(1e-8))
        else:
            # Standard softmax attention (clean_base fallback mode)
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            gate_mean = torch.tensor(0.0, device=x.device)
            gate_std = torch.tensor(0.0, device=x.device)
            actual_attn_entropy = attn_entropy

        attended = attended.transpose(1, 2).reshape(b, ca_in.shape[1], c)
        ca_input_std = ca_in.detach().float().std()
        ca_output_std = attended.detach().float().std()
        attended = self.out_proj(attended)
        if ca_is_dwt:
            attended = attended.float()
            n_hf = hf_h * hf_w
            if self.dwt_ll_route_alpha > 0.0:
                # 630 Remote T2: Soft LL Route — LL 残差注入 style
                ll_out = attended[:, :n_hf, :].permute(0, 2, 1).reshape(b, c, hf_h, hf_w)
                lh_out = attended[:, n_hf:2*n_hf, :].permute(0, 2, 1).reshape(b, c, hf_h, hf_w)
                hl_out = attended[:, 2*n_hf:3*n_hf, :].permute(0, 2, 1).reshape(b, c, hf_h, hf_w)
                hh_out = attended[:, 3*n_hf:, :].permute(0, 2, 1).reshape(b, c, hf_h, hf_w)
                ll_final = ll_f + self.dwt_ll_route_alpha * ll_out
                attended_2d = idwt2_haar(ll_final, lh_out, hl_out, hh_out).to(dtype=x.dtype)
            else:
                lh_out = attended[:, :n_hf, :].permute(0, 2, 1).reshape(b, c, hf_h, hf_w)
                hl_out = attended[:, n_hf:2*n_hf, :].permute(0, 2, 1).reshape(b, c, hf_h, hf_w)
                hh_out = attended[:, 2*n_hf:, :].permute(0, 2, 1).reshape(b, c, hf_h, hf_w)
                # 630 Phase 72 方案 C: AdaLN-Zero on LL (global_tone 调制 LL 色调)
                # LL_new = LL + γ(S_global)⊙Norm(LL) + β(S_global), γ/β 零初始化
                if self.ll_adaln_zero and global_tone is not None:
                    normed_ll = self.adaln_norm(ll_f)
                    gamma_beta = self.adaln_proj(global_tone.float()).to(dtype=ll_f.dtype)
                    gamma, beta = gamma_beta.chunk(2, dim=-1)
                    ll_f = ll_f + gamma[:, :, None, None] * normed_ll + beta[:, :, None, None]
                # 630 Phase 72 方案 D: Direct Tone Bias (无 GroupNorm, 强制 scale+shift 注入)
                # LL = LL * (1 + α*γ) + α*β, γ/β 从 global_tone 投影, α 可学习 init=0.1
                # 无 GroupNorm 内容归一化, 模型无法抑制, 必须学习风格色调注入
                if self.ll_tone_bias and global_tone is not None:
                    gamma_beta = self.tone_proj(global_tone.float()).to(dtype=ll_f.dtype)
                    gamma, beta = gamma_beta.chunk(2, dim=-1)
                    alpha = torch.tanh(self.tone_alpha)  # 限制在 [-1, 1] 防止发散
                    ll_f = ll_f * (1.0 + alpha * gamma[:, :, None, None]) + alpha * beta[:, :, None, None]
                # LL bypass: 完全保留内容锚, 仅高频子带注入 style (4J.1 原始设计)
                attended_2d = idwt2_haar(ll_f, lh_out, hl_out, hh_out).to(dtype=x.dtype)
        else:
            attended_2d = attended.transpose(1, 2).reshape(b, c, h, w)
        gate_val = self._effective_gate_value()
        style_delta = gate_val.to(dtype=x.dtype) * attended_2d
        self.pixel_entropy = pixel_entropy
        self.cross_attn_entropy = attn_entropy
        self.cross_attn_guidance = style_delta.detach().float().abs().mean(dim=1, keepdim=True).to(dtype=x.dtype)

        # Apply shortcut alpha (float only — learnable variant removed)
        alpha = self.shortcut_alpha
        x = alpha * x + style_delta

        # --- FFN ---
        x = x + self.ffn(self.norm2(x))

        # --- Debug ---
        self.last_debug = {
            "style_gate_value": torch.tanh(self.style_gate).detach().abs(),
            "cross_attn_entropy": attn_entropy.detach(),
            "actual_attn_entropy": actual_attn_entropy.detach() if isinstance(actual_attn_entropy, torch.Tensor) else torch.tensor(0.0, device=x.device),
            "gate_mean": gate_mean.detach() if isinstance(gate_mean, torch.Tensor) else torch.tensor(0.0, device=x.device),
            "gate_std": gate_std.detach() if isinstance(gate_std, torch.Tensor) else torch.tensor(0.0, device=x.device),
            "cross_attn_delta_abs": style_delta.detach().float().abs().mean(),
            "cross_attn_token_count": torch.tensor(float(style_tokens.shape[1]), device=x.device),
            "sa_input_std": h_time.detach().float().std(),
            "sa_output_std": sa_out.detach().float().std(),
            "ca_input_std": ca_input_std,
            "ca_output_std": ca_output_std,
        }
        return x

    def _attention_stats(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        h: int,
        w: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            q_stat = q.detach().float()
            k_stat = k.detach().float()
            total_queries = int(q_stat.shape[-2])
            if total_queries <= self.max_entropy_queries:
                q_small = q_stat
                sample_idx = None
            else:
                sample_idx = torch.linspace(
                    0,
                    total_queries - 1,
                    steps=self.max_entropy_queries,
                    device=q_stat.device,
                ).round().long()
                q_small = q_stat.index_select(dim=2, index=sample_idx)
            logits_small = torch.matmul(q_small, k_stat.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
            attn_small = torch.softmax(logits_small, dim=-1)
            entropy_small = -(attn_small * attn_small.clamp_min(1e-8).log()).sum(dim=-1)
            attn_entropy = entropy_small.mean().to(device=q.device, dtype=dtype)
            pixel_entropy = torch.zeros(q.shape[0], 1, h * w, device=q.device, dtype=dtype)
            if sample_idx is None:
                per_query = entropy_small.mean(dim=1, keepdim=True).to(device=q.device, dtype=dtype)
                pixel_entropy = per_query
            else:
                per_query_small = entropy_small.mean(dim=1).to(device=q.device, dtype=dtype)
                pixel_entropy.scatter_(dim=2, index=sample_idx.view(1, 1, -1).expand(q.shape[0], 1, -1), src=per_query_small.unsqueeze(1))
                pixel_entropy = F.interpolate(pixel_entropy.view(q.shape[0], 1, 1, h * w), size=(1, h * w), mode="nearest")
            return attn_entropy, pixel_entropy.view(q.shape[0], 1, h, w)


def sinusoidal_time_embedding_620(t: torch.Tensor, dim: int) -> torch.Tensor:
    t = t.float().view(-1, 1)
    half = max(1, dim // 2)
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1)))
    emb = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]
