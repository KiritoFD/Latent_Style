#!/usr/bin/env python3
"""Probe: Spectral ODE Bridge 全局瓶颈诊断 (结合 MODEL.md 理论).

理论框架 (MODEL.md + progress.json):
  - 模型是 "learned endpoint corrector", 非高精度 ODE
  - 当前瓶颈: velocity field accuracy (非 solver 精度)
  - style_gain = endpoint_pressure * semantic_style_activation * delivered_residual_amplitude
  - content_preservation = kinetic_pressure * skip_retention * routing_smoothness

诊断维度:
  A. Velocity field accuracy (核心瓶颈)
     - cosine sim(v_subband, target_delta_subband) @ various t
     - ||v|| / ||target_delta|| amplitude ratio
     - per-subband accuracy ranking (LL/LH/HL)
  B. AdaIN effect (delivered_residual_amplitude)
     - before/after: mean/std/cov distance to style
     - 协方差修正率 (AdaIN 只做 mean+std, 缺 cov)
  C. Frequency band energy
     - per-subband energy in content/target/output
     - HH 保留度 (HH 无 velocity head, 仅 IDWT 传递)
  D. Style injection health (semantic_style_activation)
     - cross-attn entropy
     - style sensitivity (shuffle style_id -> v change)
  E. Trajectory analysis (kinetic_pressure, routing_smoothness)
     - per-step distance to source/target
     - velocity contribution vs endpoint drift
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import load_experiment_config  # noqa: E402
from model import build_model_from_config  # noqa: E402
from spectral620 import dwt2_haar, idwt2_haar, dwt2_lowpass  # noqa: E402
from utils.dataset import AdaCUTLatentDataset  # noqa: E402
from utils.training import strip_compile_prefix  # noqa: E402


def _load_checkpoint_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state: dict[str, torch.Tensor] | None = None
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
        if state is None and all(isinstance(k, str) for k in payload.keys()):
            state = payload
    if state is None:
        raise TypeError(f"Unsupported checkpoint payload: {checkpoint}")
    return strip_compile_prefix({str(k): v for k, v in state.items()})


def _build_model(cfg, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge, use_checkpointing=False)
    state_dict = _load_checkpoint_state(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(json.dumps({
            "warning": "non_strict_checkpoint_load",
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
        }, ensure_ascii=False), file=sys.stderr)
    model = model.to(device=device)
    model.eval()
    return model


def _build_dataset(cfg, device: str, data_root_override: str = "") -> AdaCUTLatentDataset:
    data_cfg = cfg.data
    train_cfg = cfg.training
    return AdaCUTLatentDataset(
        data_root=str(data_root_override or data_cfg.data_root),
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=False,
        identity_ratio=data_cfg.identity_ratio,
        batch_size_hint=int(train_cfg.batch_size),
        balance_target_styles_per_batch=bool(data_cfg.balance_target_styles_per_batch),
        preload_to_gpu=False,
        preload_max_vram_gb=0.0,
        preload_reserve_ratio=float(data_cfg.preload_reserve_ratio),
        virtual_length_multiplier=float(data_cfg.virtual_length_multiplier),
        content_style_sampling_weights=data_cfg.content_style_sampling_weights,
        target_style_sampling_weights=data_cfg.target_style_sampling_weights,
        pairing_cache_path=str(data_cfg.pairing_cache_path),
        pairing_cache_topk=int(data_cfg.pairing_cache_topk),
        pairing_cache_active_topk=int(data_cfg.pairing_cache_active_topk),
        pairing_cache_sample_mode=str(data_cfg.pairing_cache_sample_mode),
        pairing_cache_rank_schedule=str(data_cfg.pairing_cache_rank_schedule),
        pairing_cache_min_topk=int(data_cfg.pairing_cache_min_topk),
        pairing_cache_curriculum_epochs=int(data_cfg.pairing_cache_curriculum_epochs),
        pairing_cache_rank_power=float(data_cfg.pairing_cache_rank_power),
        pairing_cache_explore_prob=float(data_cfg.pairing_cache_explore_prob),
        pairing_cache_explore_topk=int(data_cfg.pairing_cache_explore_topk),
        pairing_cache_dual_target_mix=float(data_cfg.pairing_cache_dual_target_mix),
        pairing_cache_dual_target_topk=int(data_cfg.pairing_cache_dual_target_topk),
        pairing_cache_aux_target_topk=int(data_cfg.pairing_cache_aux_target_topk),
        pairing_cache_cross_only=bool(data_cfg.pairing_cache_cross_only),
        latent_cache_mode=str(data_cfg.latent_cache_mode),
        latent_cache_dir=str(data_cfg.latent_cache_dir),
        style_caption_path=str(getattr(data_cfg, "style_caption_path", "")),
        device=device,
    )


def _rms(t: torch.Tensor) -> float:
    return float(t.detach().float().square().mean().sqrt().item())


def _mean_abs(t: torch.Tensor) -> float:
    return float(t.detach().float().abs().mean().item())


def _cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().float().reshape(a.shape[0], -1)
    bv = b.detach().float().reshape(b.shape[0], -1)
    return float(F.cosine_similarity(av, bv, dim=1, eps=1e-8).mean().item())


def _l2_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().float().reshape(a.shape[0], -1)
    bv = b.detach().float().reshape(b.shape[0], -1)
    return float(torch.norm(av - bv, dim=1).mean().item())


def _stats(t: torch.Tensor) -> dict[str, float]:
    """Return mean/std/cov stats for a (B,C,H,W) tensor."""
    f = t.detach().float()
    mean = f.mean(dim=[2, 3])  # [B, C]
    std = f.std(dim=[2, 3])  # [B, C]
    flat = f.reshape(f.shape[0], f.shape[1], -1)  # [B, C, HW]
    centered = flat - mean.unsqueeze(2)
    N = f.shape[2] * f.shape[3]
    cov = (centered @ centered.transpose(1, 2)) / max(N - 1, 1)  # [B, C, C]
    return {
        "mean_abs": float(mean.abs().mean().item()),
        "std_mean": float(std.mean().item()),
        "cov_frobenius": float(cov.norm(dim=[1, 2]).mean().item()),
        "cov_diag_mean": float(torch.diagonal(cov, dim1=1, dim2=2).mean().item()),
        "cov_offdiag_mean": float((cov - torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))).abs().mean().item()),
    }


def _stats_distance(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Distance between stats of a and b (B,C,H,W)."""
    sa = _stats(a)
    sb = _stats(b)
    fa = a.detach().float()
    fb = b.detach().float()
    mean_a = fa.mean(dim=[2, 3])
    mean_b = fb.mean(dim=[2, 3])
    std_a = fa.std(dim=[2, 3])
    std_b = fb.std(dim=[2, 3])
    flat_a = fa.reshape(fa.shape[0], fa.shape[1], -1)
    flat_b = fb.reshape(fb.shape[0], fb.shape[1], -1)
    ca = flat_a - mean_a.unsqueeze(2)
    cb = flat_b - mean_b.unsqueeze(2)
    N = fa.shape[2] * fa.shape[3]
    cov_a = (ca @ ca.transpose(1, 2)) / max(N - 1, 1)
    cov_b = (cb @ cb.transpose(1, 2)) / max(N - 1, 1)
    return {
        "mean_l1": float((mean_a - mean_b).abs().mean().item()),
        "std_l1": float((std_a - std_b).abs().mean().item()),
        "cov_frob": float((cov_a - cov_b).norm(dim=[1, 2]).mean().item()),
        "cov_diag_l1": float((torch.diagonal(cov_a, dim1=1, dim2=2) - torch.diagonal(cov_b, dim1=1, dim2=2)).abs().mean().item()),
        "cov_offdiag_l1": float(((cov_a - cov_b) - torch.diag_embed(torch.diagonal(cov_a - cov_b, dim1=1, dim2=2))).abs().mean().item()),
    }


# ============ Probe A: Velocity field accuracy ============
def probe_a_velocity_accuracy(
    model, content, target, target_style_id, style_latent, device, n_samples=4,
) -> dict[str, Any]:
    """诊断 velocity field 精度 (核心瓶颈).

    理论: 训练目标 v ≈ dwt(target - content), 但推理时 x_t = (1-t)*content + t*target.
    所以 ideal velocity at t: v_ideal = (target - x_t) / (1-t) = (target - content) (when t-independent).
    但模型学的是 FM target: target_delta = target - content, 然后分频.
    """
    print("\n" + "=" * 70)
    print("PROBE A: Velocity field accuracy (核心瓶颈诊断)")
    print("=" * 70)
    results: dict[str, Any] = {"per_t": [], "per_subband_summary": {}}

    target_delta = target - content
    td_ll, td_lh, td_hl, td_hh = dwt2_haar(target_delta)
    print(f"target_delta subband RMS — LL: {_rms(td_ll):.4f}, LH: {_rms(td_lh):.4f}, "
          f"HL: {_rms(td_hl):.4f}, HH: {_rms(td_hh):.4f}")
    results["target_delta_rms"] = {"ll": _rms(td_ll), "lh": _rms(td_lh), "hl": _rms(td_hl), "hh": _rms(td_hh)}

    t_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    print(f"\n{'t':>6} | {'sub':>4} | {'cos(v,td)':>10} | {'||v||/||td||':>13} | {'||v||':>8} | {'||td||':>8}")
    print("-" * 65)

    cos_acc = {"ll": [], "lh": [], "hl": []}
    amp_acc = {"ll": [], "lh": [], "hl": []}

    for t_val in t_values:
        t = torch.full((content.shape[0],), t_val, device=device, dtype=content.dtype)
        x_t = (1.0 - t_val) * content + t_val * target
        with torch.no_grad():
            v_dict = model(x_t, t=t, style_id=target_style_id, style_latent=style_latent)
        for sub_name, v_sub, td_sub in [
            ("LL", v_dict["ll"], td_ll),
            ("LH", v_dict["lh"], td_lh),
            ("HL", v_dict["hl"], td_hl),
        ]:
            cos = _cosine_flat(v_sub, td_sub)
            v_norm = _rms(v_sub)
            td_norm = _rms(td_sub)
            amp_ratio = v_norm / max(td_norm, 1e-8)
            cos_acc[sub_name.lower()].append(cos)
            amp_acc[sub_name.lower()].append(amp_ratio)
            print(f"{t_val:>6.2f} | {sub_name:>4} | {cos:>10.4f} | {amp_ratio:>13.4f} | {v_norm:>8.4f} | {td_norm:>8.4f}")
        results["per_t"].append({
            "t": t_val,
            "cos_ll": cos_acc["ll"][-1], "cos_lh": cos_acc["lh"][-1], "cos_hl": cos_acc["hl"][-1],
            "amp_ll": amp_acc["ll"][-1], "amp_lh": amp_acc["lh"][-1], "amp_hl": amp_acc["hl"][-1],
        })

    # Summary
    for sub in ["ll", "lh", "hl"]:
        results["per_subband_summary"][sub] = {
            "cos_mean": float(sum(cos_acc[sub]) / len(cos_acc[sub])),
            "cos_mid_t": cos_acc[sub][3],  # t=0.5
            "amp_mean": float(sum(amp_acc[sub]) / len(amp_acc[sub])),
            "amp_mid_t": amp_acc[sub][3],
        }
    print("\n--- Summary (velocity accuracy) ---")
    for sub in ["ll", "lh", "hl"]:
        s = results["per_subband_summary"][sub]
        print(f"  {sub.upper()}: cos_mean={s['cos_mean']:.4f}, amp_mean={s['amp_mean']:.4f} "
              f"(cos>0.5=good, amp~1.0=ideal)")
    return results


# ============ Probe B: AdaIN effect ============
def probe_b_adain_effect(
    model, content, target, target_style_id, style_latent, device,
) -> dict[str, Any]:
    """诊断 AdaIN 效果 (delivered_residual_amplitude).

    理论: AdaIN 只匹配 mean+std (对角协方差), 缺失通道间相关性.
    WCT 匹配完整协方差, 但 alpha=0.85 对 lpips 过于激进.
    """
    print("\n" + "=" * 70)
    print("PROBE B: AdaIN effect (delivered_residual_amplitude)")
    print("=" * 70)
    results: dict[str, Any] = {}

    # Run full integrate_transport to get pre-AdaIN state
    # We hijack: run ODE without AdaIN, then compare with AdaIN
    mcfg = getattr(model, 'model_cfg', None)
    endpoint_adain_scale = float(getattr(mcfg, 'endpoint_adain_scale', 0.0))
    adain_mode = str(getattr(mcfg, 'endpoint_adain_mode', 'spatial_fiber')).lower()
    print(f"Config: adain_scale={endpoint_adain_scale}, mode={adain_mode}")

    # Step 1: Run ODE without AdaIN (temporarily disable)
    original_scale = endpoint_adain_scale
    setattr(mcfg, 'endpoint_adain_scale', 0.0)
    with torch.no_grad():
        h_no_adain = model.integrate_transport(
            content, style_id=target_style_id, num_steps=8,
            style_latent=style_latent,
        )
    setattr(mcfg, 'endpoint_adain_scale', original_scale)

    # Step 2: Run ODE with AdaIN (normal)
    with torch.no_grad():
        h_with_adain = model.integrate_transport(
            content, style_id=target_style_id, num_steps=8,
            style_latent=style_latent,
        )

    # Step 3: Compare stats
    print(f"\nPre-AdaIN (ODE only) vs Post-AdaIN (ODE+AdaIN):")
    print(f"  h_no_adain RMS: {_rms(h_no_adain):.4f}")
    print(f"  h_with_adain RMS: {_rms(h_with_adain):.4f}")
    print(f"  target RMS: {_rms(target):.4f}")
    print(f"  content RMS: {_rms(content):.4f}")

    # Stats distance to style_latent
    d_no_adain = _stats_distance(h_no_adain, style_latent)
    d_with_adain = _stats_distance(h_with_adain, style_latent)
    d_target = _stats_distance(target, style_latent)

    print(f"\nStats distance to style_latent (lower=closer to style):")
    print(f"  {'metric':>16} | {'no_adain':>10} | {'with_adain':>10} | {'target':>10} | {'corr_rate':>10}")
    print("  " + "-" * 70)
    for key in d_no_adain:
        corr = (d_no_adain[key] - d_with_adain[key]) / max(d_no_adain[key], 1e-8) * 100
        print(f"  {key:>16} | {d_no_adain[key]:>10.4f} | {d_with_adain[key]:>10.4f} | {d_target[key]:>10.4f} | {corr:>9.1f}%")
    results["stats_distance_to_style"] = {
        "no_adain": d_no_adain, "with_adain": d_with_adain, "target": d_target,
    }

    # Per-subband analysis
    print(f"\nPer-subband energy (RMS):")
    for name, h in [("content", content), ("target", target), ("style", style_latent),
                     ("no_adain", h_no_adain), ("with_adain", h_with_adain)]:
        ll, lh, hl, hh = dwt2_haar(h)
        print(f"  {name:>12}: LL={_rms(ll):.4f}, LH={_rms(lh):.4f}, HL={_rms(hl):.4f}, HH={_rms(hh):.4f}")

    # AdaIN delta per subband
    delta_adain = h_with_adain - h_no_adain
    d_ll, d_lh, d_hl, d_hh = dwt2_haar(delta_adain)
    print(f"\nAdaIN delta per subband (what AdaIN actually changed):")
    print(f"  LL={_rms(d_ll):.4f}, LH={_rms(d_lh):.4f}, HL={_rms(d_hl):.4f}, HH={_rms(d_hh):.4f}")
    results["adain_delta_rms"] = {"ll": _rms(d_ll), "lh": _rms(d_lh), "hl": _rms(d_hl), "hh": _rms(d_hh)}

    # LPIPS-relevant: content preservation
    content_l2_no = _l2_flat(h_no_adain, content)
    content_l2_with = _l2_flat(h_with_adain, content)
    target_l2_no = _l2_flat(h_no_adain, target)
    target_l2_with = _l2_flat(h_with_adain, target)
    print(f"\nL2 distance (latent space):")
    print(f"  to content: no_adain={content_l2_no:.4f} -> with_adain={content_l2_with:.4f} (delta={content_l2_with-content_l2_no:+.4f})")
    print(f"  to target:  no_adain={target_l2_no:.4f} -> with_adain={target_l2_with:.4f} (delta={target_l2_with-target_l2_no:+.4f})")
    results["l2_distance"] = {
        "to_content": {"no_adain": content_l2_no, "with_adain": content_l2_with},
        "to_target": {"no_adain": target_l2_no, "with_adain": target_l2_with},
    }
    return results


# ============ Probe C: Frequency band energy ============
def probe_c_frequency_energy(
    model, content, target, target_style_id, style_latent, device,
) -> dict[str, Any]:
    """诊断频域能量分布."""
    print("\n" + "=" * 70)
    print("PROBE C: Frequency band energy distribution")
    print("=" * 70)
    results: dict[str, Any] = {}

    with torch.no_grad():
        output = model.integrate_transport(
            content, style_id=target_style_id, num_steps=8,
            style_latent=style_latent,
        )

    for name, h in [("content", content), ("target", target), ("output", output)]:
        ll, lh, hl, hh = dwt2_haar(h)
        total_e = _rms(ll) ** 2 + _rms(lh) ** 2 + _rms(hl) ** 2 + _rms(hh) ** 2
        results[name] = {
            "ll": _rms(ll), "lh": _rms(lh), "hl": _rms(hl), "hh": _rms(hh),
            "ll_frac": _rms(ll) ** 2 / max(total_e, 1e-8),
            "lh_frac": _rms(lh) ** 2 / max(total_e, 1e-8),
            "hl_frac": _rms(hl) ** 2 / max(total_e, 1e-8),
            "hh_frac": _rms(hh) ** 2 / max(total_e, 1e-8),
        }
        print(f"  {name:>8}: LL={_rms(ll):.4f} ({results[name]['ll_frac']*100:.1f}%), "
              f"LH={_rms(lh):.4f} ({results[name]['lh_frac']*100:.1f}%), "
              f"HL={_rms(hl):.4f} ({results[name]['hl_frac']*100:.1f}%), "
              f"HH={_rms(hh):.4f} ({results[name]['hh_frac']*100:.1f}%)")

    # HH preservation (HH has no velocity head, only IDWT passthrough)
    out_ll, out_lh, out_hl, out_hh = dwt2_haar(output)
    tgt_ll, tgt_lh, tgt_hl, tgt_hh = dwt2_haar(target)
    cnt_ll, cnt_lh, cnt_hl, cnt_hh = dwt2_haar(content)
    print(f"\nHH analysis (no velocity head — should equal content's HH):")
    print(f"  HH_content: {_rms(cnt_hh):.4f}, HH_output: {_rms(out_hh):.4f}, HH_target: {_rms(tgt_hh):.4f}")
    print(f"  HH output-content L2: {_l2_flat(out_hh, cnt_hh):.4f}")
    print(f"  HH output-target L2: {_l2_flat(out_hh, tgt_hh):.4f}")
    results["hh_preservation"] = {
        "content_rms": _rms(cnt_hh), "output_rms": _rms(out_hh), "target_rms": _rms(tgt_hh),
        "output_to_content_l2": _l2_flat(out_hh, cnt_hh),
        "output_to_target_l2": _l2_flat(out_hh, tgt_hh),
    }
    return results


# ============ Probe D: Style injection health ============
def probe_d_style_injection(
    model, content, target, target_style_id, style_latent, device,
) -> dict[str, Any]:
    """诊断风格注入健康度 (semantic_style_activation)."""
    print("\n" + "=" * 70)
    print("PROBE D: Style injection health (semantic_style_activation)")
    print("=" * 70)
    results: dict[str, Any] = {}

    t = torch.full((content.shape[0],), 0.5, device=device, dtype=content.dtype)
    x_t = 0.5 * content + 0.5 * target

    # 1. Cross-attention entropy
    with torch.no_grad():
        _ = model(x_t, t=t, style_id=target_style_id, style_latent=style_latent)
    cross_entropy = float(model.last_cross_attn_entropy.item()) if hasattr(model, 'last_cross_attn_entropy') else 0.0
    print(f"Cross-attn entropy: {cross_entropy:.4f}")
    print(f"  (high entropy = diffuse attention = weak style signal)")

    # 2. Style sensitivity: shuffle style_id, measure velocity change
    with torch.no_grad():
        v_orig = model(x_t, t=t, style_id=target_style_id, style_latent=style_latent)

    # Shuffle style_id
    perm = torch.randperm(target_style_id.shape[0], device=device)
    shuffled_id = target_style_id[perm]
    with torch.no_grad():
        v_shuffled = model(x_t, t=t, style_id=shuffled_id, style_latent=style_latent)

    print(f"\nStyle sensitivity (shuffle style_id -> velocity change):")
    for sub in ["ll", "lh", "hl"]:
        v1 = v_orig[sub]
        v2 = v_shuffled[sub]
        cos = _cosine_flat(v1, v2)
        delta = _mean_abs(v1 - v2)
        norm = _mean_abs(v1)
        print(f"  {sub.upper()}: cos(v, v_shuffled)={cos:.4f}, |Δv|/|v|={delta/max(norm, 1e-8):.4f}")
        results[f"style_sensitivity_{sub}"] = {"cos": cos, "delta_over_norm": delta / max(norm, 1e-8)}

    results["cross_attn_entropy"] = cross_entropy

    # 3. Style token utilization (variance across token dimension)
    style_tokens, style_global = model.style_conditioner(
        style_id=target_style_id, batch=content.shape[0], device=device, dtype=content.dtype,
    )
    if torch.is_tensor(style_tokens):
        # style_tokens: [B, num_tokens, dim]
        token_std = style_tokens.std(dim=1).mean().item()  # std across tokens, avg over batch
        token_norm = style_tokens.norm(dim=-1).mean().item()
        print(f"\nStyle token utilization:")
        print(f"  shape: {list(style_tokens.shape)}")
        print(f"  token_std (across tokens): {token_std:.4f}")
        print(f"  token_norm: {token_norm:.4f}")
        results["style_token_std"] = token_std
        results["style_token_norm"] = token_norm
    return results


# ============ Probe E: Trajectory analysis ============
def probe_e_trajectory(
    model, content, target, target_style_id, style_latent, device, num_steps=8,
) -> dict[str, Any]:
    """诊断轨迹 (kinetic_pressure, routing_smoothness)."""
    print("\n" + "=" * 70)
    print("PROBE E: Trajectory analysis (kinetic_pressure, routing_smoothness)")
    print("=" * 70)
    results: dict[str, Any] = {"per_step": []}

    h = content.clone()
    mcfg = getattr(model, 'model_cfg', None)
    solver_type = str(getattr(mcfg, 'solver_type', 'euler')).lower()
    time_schedule = str(getattr(mcfg, 'time_schedule', 'linear')).lower()
    endpoint_adain_scale = float(getattr(mcfg, 'endpoint_adain_scale', 0.0))
    only_last_step = bool(getattr(mcfg, 'endpoint_adain_only_last_step', False))
    print(f"Config: solver={solver_type}, schedule={time_schedule}, "
          f"alpha={endpoint_adain_scale}, only_last_step={only_last_step}")

    import math
    def _schedule(s):
        if time_schedule == "cosine":
            return (1.0 - math.cos(math.pi * s)) / 2.0
        elif time_schedule == "warp_cos":
            p = max(0.1, float(getattr(mcfg, 'time_schedule_warp', 1.0)))
            return (1.0 - math.cos(math.pi * (s ** p))) / 2.0
        return s

    dt = 1.0 / num_steps
    print(f"\n{'step':>4} | {'t':>6} | {'d_to_src':>10} | {'d_to_tgt':>10} | {'||v_ll||':>9} | {'||v_lh||':>9} | {'||v_hl||':>9} | {'step_Δ':>9}")
    print("-" * 85)

    for i in range(num_steps):
        t_curr = _schedule(float(i) / num_steps)
        t_next = _schedule(float(i + 1) / num_steps)
        t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)

        h_before = h.clone()
        with torch.no_grad():
            v_dict = model(h, t=t_batch, style_id=target_style_id, style_latent=style_latent)

        # Save velocity norms before integration
        v_ll_norm = _rms(v_dict["ll"])
        v_lh_norm = _rms(v_dict["lh"])
        v_hl_norm = _rms(v_dict["hl"])

        # Euler step (manually, to track pre-AdaIN state)
        ll, lh, hl, hh = dwt2_haar(h)
        ll_new = ll + v_dict["ll"] * dt
        lh_new = lh + v_dict["lh"] * dt
        hl_new = hl + v_dict["hl"] * dt
        h_next_raw = idwt2_haar(ll_new, lh_new, hl_new, hh)

        # Apply AdaIN if last step
        if only_last_step and i == num_steps - 1 and endpoint_adain_scale > 0.0:
            with torch.no_grad():
                h_next = model.integrate_transport(
                    content, style_id=target_style_id, num_steps=num_steps,
                    style_latent=style_latent,
                )
                # Just use full integration for last step display
        else:
            h_next = h_next_raw

        d_src = _l2_flat(h_next, content)
        d_tgt = _l2_flat(h_next, target)
        step_delta = _rms(h_next - h_before)

        print(f"{i+1:>4} | {t_curr:>6.3f} | {d_src:>10.4f} | {d_tgt:>10.4f} | "
              f"{v_ll_norm:>9.4f} | {v_lh_norm:>9.4f} | {v_hl_norm:>9.4f} | {step_delta:>9.4f}")
        results["per_step"].append({
            "step": i + 1, "t_curr": t_curr, "t_next": t_next,
            "d_to_src": d_src, "d_to_tgt": d_tgt,
            "v_ll": v_ll_norm, "v_lh": v_lh_norm, "v_hl": v_hl_norm,
            "step_delta": step_delta,
        })
        h = h_next

    # Final analysis
    final_d_src = _l2_flat(h, content)
    final_d_tgt = _l2_flat(h, target)
    ideal_d_src = _l2_flat(target, content)
    print(f"\nFinal: d_to_src={final_d_src:.4f}, d_to_tgt={final_d_tgt:.4f}")
    print(f"Ideal (target): d_to_src={ideal_d_src:.4f}, d_to_tgt=0.0000")
    print(f"Target reach ratio: {1.0 - final_d_tgt / max(ideal_d_src, 1e-8):.4f} "
          f"(1.0 = perfect, 0.0 = no movement)")
    results["final"] = {
        "d_to_src": final_d_src, "d_to_tgt": final_d_tgt,
        "ideal_d_to_src": ideal_d_src,
        "target_reach_ratio": 1.0 - final_d_tgt / max(ideal_d_src, 1e-8),
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe spectral ODE bridge global bottleneck.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--data-root-override", type=str, default="")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")

    dataset = _build_dataset(cfg, str(device), data_root_override=str(args.data_root_override or ""))
    print(f"Dataset size: {len(dataset)}")

    model = _build_model(cfg, args.checkpoint, device)
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    # Collect a batch
    n = min(args.n_samples, len(dataset))
    items = [dataset[i] for i in range(n)]
    content = torch.stack([it["content"] for it in items]).to(device)
    target = torch.stack([it["target_style"] for it in items]).to(device)
    target_style_id = torch.tensor([it["target_style_id"] for it in items], device=device, dtype=torch.long)
    # style_latent for AdaIN: use target_style (the reference style latent) when target_style_latent absent
    if "target_style_latent" in items[0]:
        style_latent = torch.stack([it["target_style_latent"] for it in items]).to(device)
    else:
        style_latent = target  # target_style IS the style reference for AdaIN
    print(f"Batch: content={list(content.shape)}, target={list(target.shape)}, "
          f"style_id={target_style_id.tolist()}, style_latent={'yes' if style_latent is not None else 'no'}")

    all_results: dict[str, Any] = {"config": str(args.config), "checkpoint": str(args.checkpoint)}

    # Run all probes
    all_results["probe_a_velocity"] = probe_a_velocity_accuracy(
        model, content, target, target_style_id, style_latent, device,
    )
    all_results["probe_b_adain"] = probe_b_adain_effect(
        model, content, target, target_style_id, style_latent, device,
    )
    all_results["probe_c_freq"] = probe_c_frequency_energy(
        model, content, target, target_style_id, style_latent, device,
    )
    all_results["probe_d_style"] = probe_d_style_injection(
        model, content, target, target_style_id, style_latent, device,
    )
    all_results["probe_e_trajectory"] = probe_e_trajectory(
        model, content, target, target_style_id, style_latent, device,
    )

    # Final diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY (结合理论)")
    print("=" * 70)
    pa = all_results["probe_a_velocity"]["per_subband_summary"]
    print(f"\n[A] Velocity accuracy (bottleneck):")
    for sub in ["ll", "lh", "hl"]:
        s = pa[sub]
        status = "GOOD" if s["cos_mean"] > 0.5 else ("WEAK" if s["cos_mean"] > 0.2 else "DEAD")
        print(f"  {sub.upper()}: cos={s['cos_mean']:.4f} ({status}), amp={s['amp_mean']:.4f}")

    pb = all_results["probe_b_adain"]
    d_no = pb["stats_distance_to_style"]["no_adain"]
    d_with = pb["stats_distance_to_style"]["with_adain"]
    print(f"\n[B] AdaIN effect:")
    for key in ["mean_l1", "std_l1", "cov_frob"]:
        corr = (d_no[key] - d_with[key]) / max(d_no[key], 1e-8) * 100
        print(f"  {key}: {corr:.1f}% corrected (no_adain={d_no[key]:.4f} -> with_adain={d_with[key]:.4f})")

    pd = all_results["probe_d_style"]
    print(f"\n[D] Style injection:")
    print(f"  cross_attn_entropy: {pd['cross_attn_entropy']:.4f}")
    for sub in ["ll", "lh", "hl"]:
        s = pd[f"style_sensitivity_{sub}"]
        print(f"  {sub.upper()} style sensitivity: cos={s['cos']:.4f}, |Δv|/|v|={s['delta_over_norm']:.4f}")

    pe = all_results["probe_e_trajectory"]["final"]
    print(f"\n[E] Trajectory:")
    print(f"  target_reach_ratio: {pe['target_reach_ratio']:.4f} (1.0=perfect)")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
