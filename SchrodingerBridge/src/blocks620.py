from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax: exact sparse version of softmax.

    Produces exact zeros, forcing the attention to focus on a subset of tokens.
    Reference: Martins & Astudillo (2016), "From Softmax to Sparsemax"

    Returns a probability distribution (sums to 1) with exact zeros.
    """
    # Move dim to last for processing
    logits = logits.transpose(dim, -1)
    original_shape = logits.shape
    z = logits.reshape(-1, original_shape[-1])

    # Sort in descending order
    z_sorted, _ = torch.sort(z, dim=-1, descending=True)

    # Compute k(z) and tau(z)
    z_cumsum = torch.cumsum(z_sorted, dim=-1)
    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    # Check condition: z_sorted > (cumsum - 1) / k
    cond = z_sorted > (z_cumsum - 1) / k
    # k(z) = number of True in cond
    k_z = cond.sum(dim=-1, keepdim=True).to(z.dtype)
    # tau(z) = (sum of top-k - 1) / k
    # Gather the top-k values
    mask = cond.float()
    # sum of selected elements
    sum_selected = (z_sorted * mask).sum(dim=-1, keepdim=True)
    tau = (sum_selected - 1) / k_z.clamp_min(1)
    # p = max(0, z - tau)
    p = torch.clamp(z - tau, min=0)
    # Reshape back
    p = p.reshape(original_shape).transpose(dim, -1)
    return p


class SpatialBridgeBlock620(nn.Module):
    """AdaLN(time) → Self-Attention → Cross-Attention(style) → FFN."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        style_gate_init: float = 0.05,
        style_gate_mode: str = "tanh_gate",
        style_moe_enabled: bool = False,
        style_moe_num_experts: int = 4,
        style_moe_router_hidden_dim: int = 128,
        style_kv_moe_content_routed: bool = False,
        style_shortcut_alpha: Any = 1.0,
        style_query_source: str = "concat",
        style_cross_attn_skip_coarse: bool = False,
        style_attn_topk: int = 0,
        layer_idx: int = 0,
        num_layers: int = 4,
        dino_dim: int = 384,
        film_enabled: bool = False,
        attn_mode: str = "softmax",
        attn_temperature: float = 1.0,
        gate_warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_layers = int(num_layers)
        self.gate_warmup_steps = max(0, int(gate_warmup_steps))
        self._current_step = 0
        self.style_query_source = str(style_query_source).strip().lower()
        self.style_cross_attn_skip_coarse = bool(style_cross_attn_skip_coarse)
        self.style_attn_topk = int(style_attn_topk)
        self.max_entropy_queries = 256
        self.cross_attn_entropy = torch.tensor(0.0)

        # Parse shortcut_alpha
        if isinstance(style_shortcut_alpha, (list, tuple)):
            if self.layer_idx < len(style_shortcut_alpha):
                self.shortcut_alpha = float(style_shortcut_alpha[self.layer_idx])
            else:
                self.shortcut_alpha = 1.0
        elif isinstance(style_shortcut_alpha, str) and style_shortcut_alpha.lower() == "learnable":
            self.shortcut_alpha = "learnable"
            self.shortcut_w = nn.Parameter(torch.tensor(2.2))
        else:
            self.shortcut_alpha = float(style_shortcut_alpha)
        self.dim = int(dim)
        self.num_heads = max(1, min(int(num_heads), self.dim))
        while self.dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.dim // self.num_heads
        self.style_moe_enabled = bool(style_moe_enabled)
        self.style_moe_num_experts = max(1, int(style_moe_num_experts))
        self.style_kv_moe_content_routed = bool(style_kv_moe_content_routed)
        self.norm1 = nn.GroupNorm(1, self.dim, affine=False)
        self.time_adaln = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, self.dim * 3))
        # Self-attention: content Q/K/V
        self.sa_qkv = nn.Linear(self.dim, self.dim * 3)
        self.sa_out = nn.Linear(self.dim, self.dim)
        # Cross-attention: content Q, style K/V
        self.q_proj = nn.Linear(self.dim, self.dim)
        if self.style_query_source == "content_dino":
            self.q_dino_proj = nn.Linear(int(dino_dim), self.dim)
        if self.style_moe_enabled:
            router_hidden = max(1, int(style_moe_router_hidden_dim))
            router_in_dim = self.dim * 2 if self.style_kv_moe_content_routed else self.dim
            self.style_moe_router = nn.Sequential(
                nn.LayerNorm(router_in_dim),
                nn.Linear(router_in_dim, router_hidden),
                nn.SiLU(),
                nn.Linear(router_hidden, self.style_moe_num_experts),
            )
            self.k_proj_experts = nn.ModuleList(nn.Linear(self.dim, self.dim) for _ in range(self.style_moe_num_experts))
            self.v_proj_experts = nn.ModuleList(nn.Linear(self.dim, self.dim) for _ in range(self.style_moe_num_experts))
        else:
            self.k_proj = nn.Linear(self.dim, self.dim)
            self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.norm2 = nn.GroupNorm(1, self.dim, affine=False)
        self.ffn = nn.Sequential(
            nn.GroupNorm(1, self.dim),
            nn.Conv2d(self.dim, self.dim * 4, kernel_size=1),  # 4x expansion (was 2x)
            nn.SiLU(),
            nn.Conv2d(self.dim * 4, self.dim, kernel_size=1),
        )
        self.style_gate = nn.Parameter(torch.tensor(float(style_gate_init)))
        self.style_gate_mode = str(style_gate_mode).strip().lower()
        self._gate_init = float(style_gate_init)
        self.film_enabled = bool(film_enabled)
        self.attn_mode = str(attn_mode).strip().lower()
        self.attn_temperature = float(attn_temperature)
        self.film_init_std = float(getattr(self, "film_init_std", 0.02))
        if self.film_enabled:
            # Post-cross-attention FiLM: modulates features after style injection
            # Non-zero init: gamma starts small but style-dependent, breaking the
            # "model ignores style" equilibrium that zero-init causes.
            self.film_proj = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.dim * 2),
            )
            nn.init.normal_(self.film_proj[-1].weight, mean=0.0, std=self.film_init_std)
            nn.init.zeros_(self.film_proj[-1].bias)
            # Pre-cross-attention FiLM: makes Q style-dependent
            self.film_q_proj = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.dim * 2),
            )
            nn.init.normal_(self.film_q_proj[-1].weight, mean=0.0, std=self.film_init_std)
            nn.init.zeros_(self.film_q_proj[-1].bias)
            # Style-conditioned attention bias: directly adds per-token bias to
            # attention logits before softmax. This bypasses the Q@K^T bottleneck
            # and provides a strong, direct style signal that the softmax can't average away.
            self.style_bias_proj = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, 256),  # one bias per style token
            )
            nn.init.normal_(self.style_bias_proj[-1].weight, mean=0.0, std=self.film_init_std)
            nn.init.zeros_(self.style_bias_proj[-1].bias)
        else:
            self.film_proj = None
            self.film_q_proj = None
            self.style_bias_proj = None
        nn.init.zeros_(self.time_adaln[-1].weight)
        nn.init.zeros_(self.time_adaln[-1].bias)
        nn.init.zeros_(self.sa_out.bias)
        nn.init.zeros_(self.out_proj.bias)
        self.last_debug: dict[str, torch.Tensor] = {}

    def set_step(self, step: int) -> None:
        """Update the current training step for gate warmup scheduling."""
        self._current_step = int(step)

    def _effective_gate_value(self) -> torch.Tensor:
        """Compute effective gate value with warmup schedule.
        
        During warmup (step < gate_warmup_steps), the gate value is linearly
        scaled from 0 to the learned parameter value. This prevents the cold-start
        problem where a fully-open gate with random features creates noise that
        gradient descent punishes by collapsing the gate.
        """
        raw = torch.tanh(self.style_gate)
        if self.gate_warmup_steps <= 0 or not self.training:
            return raw
        warmup_factor = min(1.0, self._current_step / max(1, self.gate_warmup_steps))
        return raw * warmup_factor

    def _style_kv(self, style_tokens: torch.Tensor, *, content_features: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.style_moe_enabled:
            return self.k_proj(style_tokens), self.v_proj(style_tokens), None
        style_summary = style_tokens.float().mean(dim=1).to(dtype=style_tokens.dtype)
        if self.style_kv_moe_content_routed and content_features is not None:
            content_summary = content_features.float().mean(dim=[2, 3]).to(dtype=style_tokens.dtype)
            router_in = torch.cat([content_summary, style_summary], dim=-1)
        else:
            router_in = style_summary
        router_probs = torch.softmax(self.style_moe_router(router_in).float(), dim=-1).to(dtype=style_tokens.dtype)
        k_experts = torch.stack([proj(style_tokens) for proj in self.k_proj_experts], dim=1)
        v_experts = torch.stack([proj(style_tokens) for proj in self.v_proj_experts], dim=1)
        weights = router_probs[:, :, None, None]
        k = (k_experts * weights).sum(dim=1)
        v = (v_experts * weights).sum(dim=1)
        return k, v, router_probs

    def forward(
        self,
        x: torch.Tensor,
        *,
        time_emb: torch.Tensor,
        style_tokens: torch.Tensor,
        style_global: torch.Tensor | None = None,
        content_dino_patches: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        skip_cross = self.style_cross_attn_skip_coarse and (self.layer_idx < self.num_layers // 2)
        if skip_cross:
            style_delta = torch.zeros_like(x)
            attn_entropy = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            pre_film_gamma_abs = torch.tensor(0.0, device=x.device)
            pre_film_beta_abs = torch.tensor(0.0, device=x.device)
            ca_input_std = torch.tensor(0.0, device=x.device)
            ca_output_std = torch.tensor(0.0, device=x.device)
        else:
            # Compute content features for cross-attention query
            ca_in = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

            # --- Pre-Cross-Attention FiLM: make Q style-dependent ---
            # This is the KEY fix: by modulating the content features BEFORE they are
            # projected into queries, the attention weights Q @ K^T become style-specific.
            # Without this, cross-attention produces uniform attention (entropy ~ ln(256))
            # regardless of style input, causing the model to ignore style.
            pre_film_gamma_abs = torch.tensor(0.0, device=x.device)
            pre_film_beta_abs = torch.tensor(0.0, device=x.device)
            if self.film_enabled and style_global is not None:
                film_q_params = self.film_q_proj(style_global.float()).to(dtype=x.dtype)
                gamma_q, beta_q = film_q_params.chunk(2, dim=-1)  # [B, dim] each
                ca_in = (1.0 + gamma_q[:, None, :]) * ca_in + beta_q[:, None, :]
                with torch.no_grad():
                    pre_film_gamma_abs = gamma_q.detach().float().abs().mean()
                    pre_film_beta_abs = beta_q.detach().float().abs().mean()

            if self.style_query_source == "content_dino" and content_dino_patches is not None:
                dino_b, dino_tokens, dino_c = content_dino_patches.shape
                side = int(round(dino_tokens ** 0.5))
                if side * side == dino_tokens:
                    dino_grid = content_dino_patches.reshape(dino_b, side, side, dino_c).permute(0, 3, 1, 2)
                    dino_up = F.interpolate(dino_grid.float(), size=(h, w), mode="bilinear", align_corners=False).to(dtype=x.dtype)
                    dino_flat = dino_up.permute(0, 2, 3, 1).reshape(b, h * w, dino_c)
                else:
                    dino_flat = content_dino_patches.float().repeat_interleave(max(1, (h * w) // dino_tokens), dim=1)[:, :h * w, :].to(dtype=x.dtype)
                
                q_in = self.q_dino_proj(dino_flat)
                q = q_in.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
            elif self.style_query_source == "sa_out_only":
                ca_in_sa = sa_delta.permute(0, 2, 3, 1).reshape(b, h * w, c)
                q = self.q_proj(ca_in_sa).view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
            else:
                # Default: use pre-FiLM-modulated ca_in as query input
                q = self.q_proj(ca_in).view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)

            k_tokens, v_tokens, router_probs = self._style_kv(style_tokens, content_features=x)
            k = k_tokens.view(b, style_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
            v = v_tokens.view(b, style_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

            attn_entropy, pixel_entropy = self._attention_stats(q, k, h=h, w=w, dtype=x.dtype)

            # --- Style-conditioned attention bias ---
            # Directly adds per-token bias to attention logits before softmax.
            # This bypasses the Q@K^T bottleneck: the softmax can't average away
            # an additive bias the way it can with multiplicative Q modulation.
            style_bias = None
            if self.film_enabled and self.style_bias_proj is not None and style_global is not None:
                style_bias = self.style_bias_proj(style_global.float()).to(dtype=x.dtype)  # [B, 256]
                style_bias = style_bias[:, None, None, :]  # [B, 1, 1, 256] for broadcasting

            # --- Attention computation ---
            # Modes: softmax (default), gated (sigmoid, no normalization), sparsemax
            # Temperature: <1 sharpens distribution, >1 smooths
            scale = 1.0 / math.sqrt(float(self.head_dim))
            temp = max(self.attn_temperature, 1e-4)
            gate_mean = torch.tensor(0.0, device=x.device)
            gate_std = torch.tensor(0.0, device=x.device)
            actual_attn_entropy = attn_entropy  # default: softmax entropy from _attention_stats

            if self.attn_mode == "gated":
                # Gated attention: sigmoid instead of softmax, no normalization
                # Each token independently gated, style_bias directly controls gates
                logits = torch.matmul(q, k.transpose(-2, -1)) * scale
                if style_bias is not None:
                    logits = logits + style_bias
                gates = torch.sigmoid(logits / temp)
                attended = torch.matmul(gates, v)
                # Renormalize by sum of gates for stable output scale
                gate_sum = gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                attended = attended / gate_sum
                with torch.no_grad():
                    gate_mean = gates.detach().float().mean()
                    gate_std = gates.detach().float().std()
                    # Entropy of normalized gates (treat gates as unnormalized distribution)
                    gates_norm = gates / gate_sum
                    actual_attn_entropy = -(gates_norm * gates_norm.clamp_min(1e-8).log()).sum(dim=-1).mean()
            elif self.attn_mode == "gated_raw":
                # Gated attention WITHOUT renormalization.
                # This is the key change: the output is not pulled back to the mean of V.
                # Each (query, token) pair independently scales its V contribution.
                logits = torch.matmul(q, k.transpose(-2, -1)) * scale
                if style_bias is not None:
                    logits = logits + style_bias
                gates = torch.sigmoid(logits / temp)
                attended = torch.matmul(gates, v)
                with torch.no_grad():
                    gate_mean = gates.detach().float().mean()
                    gate_std = gates.detach().float().std()
                    # Entropy-like sparsity: fraction of active gates
                    active_ratio = (gates > 0.5).float().mean()
                    actual_attn_entropy = -torch.log(active_ratio.clamp_min(1e-8))
            elif self.attn_mode == "relu2":
                # ReLU^2 attention: no softmax, sparse, magnitude-preserving.
                logits = torch.matmul(q, k.transpose(-2, -1)) * scale / temp
                if style_bias is not None:
                    logits = logits + style_bias
                gates = torch.relu(logits) ** 2
                attended = torch.matmul(gates, v)
                with torch.no_grad():
                    gate_mean = gates.detach().float().mean()
                    gate_std = gates.detach().float().std()
                    active_ratio = (gates > 0.0).float().mean()
                    actual_attn_entropy = -torch.log(active_ratio.clamp_min(1e-8))
            elif self.attn_mode == "style_select":
                # Style-global selects a subset of style tokens before attention.
                # First compute raw affinities, then style_global modulates a top-k mask.
                logits = torch.matmul(q, k.transpose(-2, -1)) * scale / temp
                if style_bias is not None:
                    logits = logits + style_bias
                # Top-k selection per query (k=16 by default, ~6% of 256 tokens)
                select_k = min(16, logits.shape[-1])
                topk_val, topk_idx = torch.topk(logits, k=select_k, dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(dim=-1, index=topk_idx, src=topk_val)
                attn = torch.softmax(mask, dim=-1)
                attended = torch.matmul(attn, v)
                with torch.no_grad():
                    gate_mean = attn.detach().float().mean()
                    gate_std = attn.detach().float().std()
                    actual_attn_entropy = -(attn * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()
            elif self.attn_mode == "sparsemax":
                # Sparsemax: exact sparse attention, produces exact zeros
                logits = torch.matmul(q, k.transpose(-2, -1)) * scale / temp
                if style_bias is not None:
                    logits = logits + style_bias
                attn = _sparsemax(logits, dim=-1)
                attended = torch.matmul(attn, v)
                with torch.no_grad():
                    actual_attn_entropy = -(attn * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()
                    gate_mean = attn.detach().float().mean()
                    gate_std = attn.detach().float().std()
            elif self.style_attn_topk > 0:
                # Top-k sparse attention with standard softmax
                logits = torch.matmul(q, k.transpose(-2, -1)) * scale / temp
                if style_bias is not None:
                    logits = logits + style_bias
                topk_val, topk_idx = torch.topk(logits, k=min(self.style_attn_topk, logits.shape[-1]), dim=-1)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(dim=-1, index=topk_idx, src=topk_val)
                attn = torch.softmax(mask, dim=-1)
                attended = torch.matmul(attn, v)
            elif self.attn_temperature != 1.0 or style_bias is not None:
                # Standard softmax with temperature and/or style_bias
                logits = torch.matmul(q, k.transpose(-2, -1)) * scale / temp
                if style_bias is not None:
                    logits = logits + style_bias
                attn = torch.softmax(logits, dim=-1)
                attended = torch.matmul(attn, v)
            else:
                # Default: fast sdpa path (no bias, no temperature)
                attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            attended = attended.transpose(1, 2).reshape(b, h * w, c)
            ca_input_std = ca_in.detach().float().std()
            ca_output_std = attended.detach().float().std()
            attended = self.out_proj(attended)
            attended_2d = attended.transpose(1, 2).reshape(b, c, h, w)
            if self.style_gate_mode == "fixed_one":
                style_delta = attended_2d
            elif self.style_gate_mode == "film_only":
                style_delta = torch.zeros_like(attended_2d)
            else:
                style_delta = self._effective_gate_value().to(dtype=x.dtype) * attended_2d
            self.pixel_entropy = pixel_entropy

        if skip_cross:
            self.pixel_entropy = torch.zeros(b, 1, h, w, device=x.device, dtype=x.dtype)

        self.cross_attn_entropy = attn_entropy

        # Apply shortcut alpha
        if isinstance(self.shortcut_alpha, float):
            alpha = self.shortcut_alpha
        elif self.shortcut_alpha == "learnable":
            alpha = torch.sigmoid(self.shortcut_w).to(dtype=x.dtype)
        else:
            alpha = 1.0

        x = alpha * x + style_delta

        # --- StyleFiLM: direct style→feature modulation ---
        # Bypasses cross-attention averaging bottleneck.
        # x' = (1 + gamma(s)) * x + beta(s), s = style_global
        # Non-zero init (std=0.02) breaks the "model ignores style" equilibrium
        # that zero-init causes. Gamma starts small but style-dependent.
        film_gamma_abs = torch.tensor(0.0, device=x.device)
        film_beta_abs = torch.tensor(0.0, device=x.device)
        if self.film_enabled and style_global is not None:
            film_params = self.film_proj(style_global.float()).to(dtype=x.dtype)
            gamma, beta = film_params.chunk(2, dim=-1)  # [B, dim] each
            x = (1.0 + gamma[:, :, None, None]) * x + beta[:, :, None, None]
            with torch.no_grad():
                film_gamma_abs = gamma.detach().float().abs().mean()
                film_beta_abs = beta.detach().float().abs().mean()

        # --- FFN ---
        x = x + self.ffn(self.norm2(x))

        # --- Debug ---
        self.last_debug = {
            "style_gate_value": self._effective_gate_value().detach().abs(),
            "cross_attn_entropy": attn_entropy.detach(),
            "actual_attn_entropy": actual_attn_entropy.detach() if isinstance(actual_attn_entropy, torch.Tensor) else torch.tensor(0.0, device=x.device),
            "gate_mean": gate_mean.detach() if isinstance(gate_mean, torch.Tensor) else torch.tensor(0.0, device=x.device),
            "gate_std": gate_std.detach() if isinstance(gate_std, torch.Tensor) else torch.tensor(0.0, device=x.device),
            "cross_attn_delta_abs": style_delta.detach().float().abs().mean(),
            "cross_attn_token_count": torch.tensor(float(style_tokens.shape[1]), device=x.device),
            "film_gamma_abs": film_gamma_abs,
            "film_beta_abs": film_beta_abs,
            "film_enabled": torch.tensor(1.0 if self.film_enabled else 0.0, device=x.device),
            "pre_film_gamma_abs": pre_film_gamma_abs,
            "pre_film_beta_abs": pre_film_beta_abs,
            "style_bias_abs": style_bias.detach().float().abs().mean() if style_bias is not None else torch.tensor(0.0, device=x.device),
            "sa_input_std": h_time.detach().float().std(),
            "sa_output_std": sa_out.detach().float().std(),
            "ca_input_std": ca_input_std,
            "ca_output_std": ca_output_std,
        }
        if not skip_cross and router_probs is not None:
            with torch.no_grad():
                probs = router_probs.detach().float()
                router_entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
                router_max = probs.max(dim=-1).values.mean()
            self.last_debug["style_moe_router_entropy"] = router_entropy
            self.last_debug["style_moe_router_max_prob"] = router_max
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
