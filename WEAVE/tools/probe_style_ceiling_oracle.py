"""Oracle / reachability probe for WEAVE style ceiling (latent-space only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import load_experiment_config  # noqa: E402
from model import build_model_from_config, _adain_match_subband, _wct_match_fiber  # noqa: E402
from style_families import prune_state_dict_for_tokenizer_family  # noqa: E402
from utils.dataset import AdaCUTLatentDataset  # noqa: E402
from utils.training import strip_compile_prefix  # noqa: E402
from wavelet import dwt2_haar, idwt2_haar  # noqa: E402


def _rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt().item())


def _mean_std(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return mu, std


def _stat_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    mu_a, std_a = _mean_std(a)
    mu_b, std_b = _mean_std(b)
    return float(((mu_a - mu_b).pow(2).mean() + (std_a - std_b).pow(2).mean()).sqrt().item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.detach().float().flatten(1),
            b.detach().float().flatten(1),
            dim=1,
        ).mean().item()
    )


def _bands(z: torch.Tensor) -> dict[str, torch.Tensor]:
    ll, lh, hl, hh = dwt2_haar(z)
    return {"ll": ll, "lh": lh, "hl": hl, "hh": hh, "full": z}


def _dist(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    return {"rms": _rms(a - b), "stat_l2": _stat_l2(a, b), "cos": _cos(a, b)}


def _acc(store: dict[str, list[float]], key: str, value: float) -> None:
    store.setdefault(key, []).append(float(value))


def _mean(store: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(sum(v) / max(1, len(v))) for k, v in store.items()}


def build_dataset(config: Any, batch_size: int, data_root: Path) -> AdaCUTLatentDataset:
    ds = AdaCUTLatentDataset(
        data_root=str(data_root),
        style_subdirs=list(config.data.style_subdirs),
        allow_hflip=False,
        identity_ratio=None,
        batch_size_hint=batch_size,
        balance_target_styles_per_batch=False,
        preload_to_gpu=False,
        preload_max_vram_gb=0.0,
        preload_reserve_ratio=0.35,
        virtual_length_multiplier=1.0,
        content_style_sampling_weights=None,
        target_style_sampling_weights=None,
        pairing_cache_path="",
        pairing_cache_topk=8,
        pairing_cache_active_topk=0,
        pairing_cache_sample_mode="uniform",
        pairing_cache_rank_schedule="flat",
        pairing_cache_min_topk=1,
        pairing_cache_curriculum_epochs=0,
        pairing_cache_rank_power=1.0,
        pairing_cache_explore_prob=0.0,
        pairing_cache_explore_topk=0,
        pairing_cache_dual_target_mix=0.0,
        pairing_cache_dual_target_topk=0,
        pairing_cache_aux_target_topk=0,
        pairing_cache_cross_only=False,
        latent_cache_mode="off",
        latent_cache_dir="",
        style_caption_path="",
        device="cpu",
    )
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(0)
    return ds


def load_model(config: Any, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model_from_config(config.model, bridge_cfg=config.bridge).to(device)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = strip_compile_prefix(state)
    state, _ = prune_state_dict_for_tokenizer_family(
        state,
        tokenizer_family=str(getattr(config.model, "tokenizer_family", "legacy_factorized")),
        contract_family=str(getattr(config.model, "contract_family", "legacy")),
        style_injection_mode=str(getattr(config.model, "style_injection_mode", "none")),
        proximal_mode=str(getattr(config.model, "proximal_mode", "off")),
        style_delta_mode=str(getattr(config.model, "style_delta_mode", "none")),
        output_appearance_alignment_mode=str(getattr(config.model, "output_appearance_alignment_mode", "none")),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    model._probe_load_info = {"missing": len(missing), "unexpected": len(unexpected)}  # type: ignore[attr-defined]
    model.eval()
    return model


def set_adain_scale(model: torch.nn.Module, scale: float) -> None:
    mcfg = getattr(model, "model_cfg", None)
    if mcfg is not None and hasattr(mcfg, "endpoint_adain_scale"):
        mcfg.endpoint_adain_scale = float(scale)


def make_sat(content: torch.Tensor, style: torch.Tensor, alpha: float) -> torch.Tensor:
    ll_c, _, _, _ = dwt2_haar(content)
    ll_s, lh_s, hl_s, hh_s = dwt2_haar(style)
    if alpha > 0:
        ll = (1.0 - alpha) * ll_c + alpha * _adain_match_subband(ll_c, ll_s)
    else:
        ll = ll_c
    return idwt2_haar(ll, lh_s, hl_s, hh_s)


def swap_band(pred: torch.Tensor, style: torch.Tensor, band: str) -> torch.Tensor:
    p = _bands(pred)
    s = _bands(style)
    p[band] = s[band]
    return idwt2_haar(p["ll"], p["lh"], p["hl"], p["hh"])


def ll_app(content: torch.Tensor, style: torch.Tensor, mode: str, alpha: float) -> torch.Tensor:
    ll_c, lh_c, hl_c, hh_c = dwt2_haar(content)
    ll_s, _, _, _ = dwt2_haar(style)
    matched = _adain_match_subband(ll_c, ll_s) if mode == "adain" else _wct_match_fiber(ll_c, ll_s)
    ll = (1.0 - alpha) * ll_c + alpha * matched
    return idwt2_haar(ll, lh_c, hl_c, hh_c)


@torch.no_grad()
def integrate(model: torch.nn.Module, content: torch.Tensor, style_id: torch.Tensor, style: torch.Tensor, steps: int, scale: float) -> torch.Tensor:
    set_adain_scale(model, scale)
    out = model.integrate_transport(
        content,
        style_id=style_id,
        style_latent=style,
        num_steps=steps,
    )
    if isinstance(out, dict):
        out = out.get("latent", out.get("x"))
    return out


@torch.no_grad()
def run(model, loader, device, num_batches: int, ll_alpha: float, steps: int) -> dict[str, Any]:
    store: dict[str, list[float]] = {}
    n = 0
    scales = [0.0, 1.0, 1.5, 2.0]
    for bi, raw in enumerate(loader, start=1):
        if bi > num_batches:
            break
        content = raw["content"].to(device=device, dtype=torch.float32)
        style = raw["target_style"].to(device=device, dtype=torch.float32)
        style_id = raw["target_style_id"].to(device=device, dtype=torch.long)
        n += content.shape[0]
        cb, sb = _bands(content), _bands(style)

        for band in ("ll", "lh", "hl", "hh", "full"):
            for k, v in _dist(cb[band], sb[band]).items():
                _acc(store, f"content_vs_style/{band}/{k}", v)

        delta = style - content
        d_bands = dwt2_haar(delta)
        energies = [float(x.float().pow(2).mean().item()) for x in d_bands]
        et = max(sum(energies), 1e-12)
        for band, e in zip(("ll", "lh", "hl", "hh"), energies):
            _acc(store, f"delta_energy_share/{band}", e / et)

        sat = make_sat(content, style, ll_alpha)
        satb = _bands(sat)
        for band in ("ll", "lh", "hl", "hh", "full"):
            for k, v in _dist(satb[band], sb[band]).items():
                _acc(store, f"sat_vs_style/{band}/{k}", v)
            for k, v in _dist(satb[band], cb[band]).items():
                _acc(store, f"sat_vs_content/{band}/{k}", v)

        preds = {s: integrate(model, content, style_id, style, steps, s) for s in scales}
        for s, pred in preds.items():
            pb = _bands(pred)
            tag = "no_adain" if s == 0.0 else f"adain_{s:g}"
            for band in ("ll", "lh", "hl", "hh", "full"):
                for k, v in _dist(pb[band], sb[band]).items():
                    _acc(store, f"model_{tag}_vs_style/{band}/{k}", v)
                den = max(_stat_l2(cb[band], sb[band]), 1e-8)
                _acc(store, f"transfer_ratio_{tag}/{band}", 1.0 - _stat_l2(pb[band], sb[band]) / den)

        # residual of best practical operating point
        rem = style - preds[1.5]
        rem_b = dwt2_haar(rem)
        energies = [float(x.float().pow(2).mean().item()) for x in rem_b]
        et = max(sum(energies), 1e-12)
        p15 = _bands(preds[1.5])
        for band, e in zip(("ll", "lh", "hl", "hh"), energies):
            _acc(store, f"remain_energy_share_adain15/{band}", e / et)
            _acc(store, f"remain_stat_l2_adain15/{band}", _stat_l2(p15[band], sb[band]))

        # oracle band swaps on no-adain pred, then re-apply adain 1.5 via model path on swapped? use pure swap distances
        pred0 = preds[0.0]
        for band in ("ll", "lh", "hl", "hh"):
            swapped = swap_band(pred0, style, band)
            # apply spatial-fiber adain scale 1.5 using model helper if present
            if hasattr(model, "_apply_endpoint_adain"):
                try:
                    set_adain_scale(model, 1.5)
                    swapped = model._apply_endpoint_adain(
                        swapped,
                        style_latent=style,
                        adain_mode="spatial_fiber",
                        lowpass_levels=1,
                        adain_scale_ll=0.0,
                        adain_scale_lh=1.5,
                        adain_scale_hl=1.5,
                        adain_scale_hh=1.5,
                        endpoint_adain_scale=1.5,
                    )
                except TypeError:
                    pass
            key = f"oracle_swap_{band}_adain15"
            for k, v in _dist(swapped, style).items():
                _acc(store, f"{key}/full/{k}", v)
            den = max(_stat_l2(content, style), 1e-8)
            _acc(store, f"{key}/transfer_ratio", 1.0 - _stat_l2(swapped, style) / den)

        for mode in ("adain", "wct"):
            for alpha in (0.3, 0.5, 1.0):
                app_only = ll_app(content, style, mode, alpha)
                ll_o, _, _, _ = dwt2_haar(app_only)
                _, lh_s, hl_s, hh_s = dwt2_haar(style)
                ideal = idwt2_haar(ll_o, lh_s, hl_s, hh_s)
                for name, tensor in (
                    (f"ll_{mode}_a{alpha:g}_hf_content", app_only),
                    (f"ll_{mode}_a{alpha:g}_hf_style", ideal),
                ):
                    key = f"oracle_{name}"
                    for k, v in _dist(tensor, style).items():
                        _acc(store, f"{key}/full/{k}", v)
                    den = max(_stat_l2(content, style), 1e-8)
                    _acc(store, f"{key}/transfer_ratio", 1.0 - _stat_l2(tensor, style) / den)
                    tb = _bands(tensor)
                    _acc(store, f"{key}/ll_stat_l2", _stat_l2(tb["ll"], sb["ll"]))
                    _acc(store, f"{key}/content_full_rms", _rms(tensor - content))

        # perfect SAT upper bound: content LL + style HF, alpha train
        for alpha in (0.0, 0.3, 0.5, 1.0):
            sat_a = make_sat(content, style, alpha)
            key = f"oracle_sat_alpha_{alpha:g}"
            for k, v in _dist(sat_a, style).items():
                _acc(store, f"{key}/full/{k}", v)
            den = max(_stat_l2(content, style), 1e-8)
            _acc(store, f"{key}/transfer_ratio", 1.0 - _stat_l2(sat_a, style) / den)
            _acc(store, f"{key}/content_full_rms", _rms(sat_a - content))
            _acc(store, f"{key}/ll_stat_l2", _stat_l2(_bands(sat_a)["ll"], sb["ll"]))

    return {"num_samples": n, "metrics": _mean(store)}


def to_md(results: dict[str, Any]) -> str:
    m = results["metrics"]
    lines = [
        "# Style Ceiling Oracle Probe",
        "",
        f"Config: `{results['config']}`",
        f"Checkpoint: `{results['checkpoint']}`",
        f"Data: `{results['data_root']}`",
        f"Samples: {results['num_samples']}",
        f"Load: `{results['load_info']}`",
        "",
        "## Content-style delta energy",
        "",
        "| band | share |",
        "|---|---:|",
    ]
    for b in ("ll", "lh", "hl", "hh"):
        lines.append(f"| {b} | {m[f'delta_energy_share/{b}']:.4f} |")
    lines += [
        "",
        "## Remain after model@AdaIN=1.5",
        "",
        "| band | energy share | stat L2 |",
        "|---|---:|---:|",
    ]
    for b in ("ll", "lh", "hl", "hh"):
        lines.append(
            f"| {b} | {m[f'remain_energy_share_adain15/{b}']:.4f} | {m[f'remain_stat_l2_adain15/{b}']:.6f} |"
        )
    lines += [
        "",
        "## Transfer ratio (1 - statL2(pred,style)/statL2(content,style))",
        "",
        "| setting | full | ll | lh | hl | hh |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tag in ("no_adain", "adain_1", "adain_1.5", "adain_2"):
        lines.append(
            f"| {tag} | {m[f'transfer_ratio_{tag}/full']:.4f} | {m[f'transfer_ratio_{tag}/ll']:.4f} | "
            f"{m[f'transfer_ratio_{tag}/lh']:.4f} | {m[f'transfer_ratio_{tag}/hl']:.4f} | "
            f"{m[f'transfer_ratio_{tag}/hh']:.4f} |"
        )
    lines += [
        "",
        "## Oracle SAT alpha sweep (LL blend + exact style HF)",
        "",
        "| alpha | transfer | full rms->style | content rms | ll statL2 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for a in (0.0, 0.3, 0.5, 1.0):
        key = f"oracle_sat_alpha_{a:g}"
        lines.append(
            f"| {a:g} | {m[f'{key}/transfer_ratio']:.4f} | {m[f'{key}/full/rms']:.4f} | "
            f"{m[f'{key}/content_full_rms']:.4f} | {m[f'{key}/ll_stat_l2']:.6f} |"
        )
    lines += [
        "",
        "## Oracle LL appearance + style HF",
        "",
        "| setup | transfer | full rms | content rms | ll statL2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("adain", "wct"):
        for a in (0.3, 0.5, 1.0):
            key = f"oracle_ll_{mode}_a{a:g}_hf_style"
            lines.append(
                f"| LL-{mode} a={a:g}+HF style | {m[f'{key}/transfer_ratio']:.4f} | "
                f"{m[f'{key}/full/rms']:.4f} | {m[f'{key}/content_full_rms']:.4f} | "
                f"{m[f'{key}/ll_stat_l2']:.6f} |"
            )
    lines += [
        "",
        "## Oracle single-band swap on model(no AdaIN) then AdaIN1.5",
        "",
        "| swapped | transfer | full rms | full statL2 |",
        "|---|---:|---:|---:|",
    ]
    for b in ("ll", "lh", "hl", "hh"):
        key = f"oracle_swap_{b}_adain15"
        lines.append(
            f"| {b} | {m[f'{key}/transfer_ratio']:.4f} | {m[f'{key}/full/rms']:.4f} | {m[f'{key}/full/stat_l2']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "exp_brk_a_ll03_10ep.json")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "exp" / "dino_s_break" / "brk_a_ll03_10ep" / "epoch_0010.pt")
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "model_probe" / "style_ceiling_oracle.json")
    ap.add_argument("--data-root", type=Path, default=Path("G:/wikiart27_latents_compact/train"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-batches", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = load_experiment_config(args.config)
    ll_alpha = float(getattr(config.bridge, "ll_partial_alpha", 0.0) or 0.0)
    if not bool(getattr(config.bridge, "ll_partial_style_enabled", False)):
        ll_alpha = 0.0

    ds = build_dataset(config, args.batch_size, args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = load_model(config, args.checkpoint, device)
    out = run(model, loader, device, args.num_batches, ll_alpha, args.num_steps)
    results = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "device": str(device),
        "ll_alpha_train": ll_alpha,
        "load_info": getattr(model, "_probe_load_info", {}),
        **out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(to_md(results), encoding="utf-8")
    m = results["metrics"]
    print(f"Wrote {args.output}")
    print("delta_energy", {b: round(m[f"delta_energy_share/{b}"], 4) for b in ("ll", "lh", "hl", "hh")})
    print("remain@1.5", {b: round(m[f"remain_energy_share_adain15/{b}"], 4) for b in ("ll", "lh", "hl", "hh")})
    print(
        "transfer_full",
        {
            t: round(m[f"transfer_ratio_{t}/full"], 4)
            for t in ("no_adain", "adain_1", "adain_1.5", "adain_2")
        },
    )
    print(
        "oracle_sat_transfer",
        {a: round(m[f"oracle_sat_alpha_{a:g}/transfer_ratio"], 4) for a in (0.0, 0.3, 0.5, 1.0)},
    )
    print(
        "oracle_swap_adain15",
        {b: round(m[f"oracle_swap_{b}_adain15/transfer_ratio"], 4) for b in ("ll", "lh", "hl", "hh")},
    )


if __name__ == "__main__":
    main()
