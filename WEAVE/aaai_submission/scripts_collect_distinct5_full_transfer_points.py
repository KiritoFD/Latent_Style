"""Collect Distinct5 full and transfer-only points on the remote WSL host."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge")
DOC_ROOT = ROOT / "docs" / "experiments" / "distinct5_512_20260602"
POINTS_CSV = DOC_ROOT / "tables" / "clip_style_vs_1lpips_points.csv"
OUT_TABLE = DOC_ROOT / "tables" / "clip_style_vs_1lpips_full_transfer_points.csv"
SAMAM_ROOT = Path(
    "/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/"
    "samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag/eval_curve"
)

LANCET_SUMMARIES = {
    "Baseline e1": "distinct5_512_ema_baseline_direct_atom_residual_b44_remote/full_eval/epoch_0001/summary.json",
    "Baseline e8": "distinct5_512_ema_baseline_direct_atom_residual_b44_remote/full_eval/epoch_0008/summary.json",
    "C e2": "distinct5_512_ema_variant_c_content_guided_spatial_b44_remote/full_eval/epoch_0002/summary.json",
    "D e1": "distinct5_512_ema_variant_d_vq_content_guided_b44_remote/full_eval/epoch_0001/summary.json",
    "E e1": "distinct5_512_ema_variant_e_latent_prototype_ot_queue_b44_remote/full_eval/epoch_0001/summary.json",
    "E e3": "distinct5_512_ema_variant_e_latent_prototype_ot_queue_b44_remote/full_eval/epoch_0003/summary.json",
    "F e1": "distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3_b44_remote/full_eval/epoch_0001/summary.json",
    "H e1": "distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote/full_eval/epoch_0001/summary.json",
    "H e2": "distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote/full_eval/epoch_0002/summary.json",
    "J e1": "distinct5_512_ema_variant_j_aux_hard_swd_queue_e3_b44_remote/full_eval/epoch_0001/summary.json",
    "K e1": "distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote/full_eval/epoch_0001/summary.json",
    "L e1": "distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote/full_eval/epoch_0001/summary.json",
    "M e1": "distinct5_512_ema_variant_m_style_gated_content_router_e3_b44_remote/full_eval/epoch_0001/summary.json",
}


def mean(rows: list[dict[str, str]], key: str) -> float:
    vals = []
    for r in rows:
        v = float(r[key])
        if math.isfinite(v):
            vals.append(v)
    return sum(vals) / max(1, len(vals))


def read_full_points() -> list[dict[str, object]]:
    out = []
    with POINTS_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(
                {
                    "scope": "full",
                    "family": r["family"],
                    "label": r["label"],
                    "step_or_epoch": r["step_or_epoch"],
                    "clip_style": float(r["clip_style"]),
                    "content_lpips": float(r["content_lpips"]),
                    "one_minus_lpips": float(r["one_minus_lpips"]),
                    "train_min": float(r["train_min"]),
                    "note": r["note"],
                }
            )
    return out


def collect_lancet_transfer(full_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_label = {str(r["label"]): r for r in full_rows if r["family"] == "LANCET"}
    out = []
    for label, rel in LANCET_SUMMARIES.items():
        src = by_label.get(label)
        if src is None:
            continue
        stats = json.loads((ROOT / "exp" / rel).read_text(encoding="utf-8"))["analysis"]["style_transfer_ability"]
        lpips = float(stats["content_lpips"])
        out.append(
            {
                "scope": "transfer",
                "family": "LANCET",
                "label": label,
                "step_or_epoch": src["step_or_epoch"],
                "clip_style": float(stats["clip_style"]),
                "content_lpips": lpips,
                "one_minus_lpips": 1.0 - lpips,
                "train_min": src["train_min"],
                "note": "transfer-only from summary.analysis.style_transfer_ability",
            }
        )
    return out


def collect_samam_transfer(full_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for src in [r for r in full_rows if r["family"] == "SaMAM"]:
        step = int(str(src["step_or_epoch"]))
        metrics_path = SAMAM_ROOT / f"step_{step:06d}" / "metrics.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["src_style"] != r["tgt_style"]]
        lpips = mean(rows, "lpips")
        out.append(
            {
                "scope": "transfer",
                "family": "SaMAM",
                "label": src["label"],
                "step_or_epoch": src["step_or_epoch"],
                "clip_style": mean(rows, "clip_style"),
                "content_lpips": lpips,
                "one_minus_lpips": 1.0 - lpips,
                "train_min": src["train_min"],
                "note": "transfer-only from per-image metrics.csv",
            }
        )
    return out


def collect_noop_transfer() -> list[dict[str, object]]:
    stats = json.loads((DOC_ROOT / "no_op_identity_5x5" / "summary.json").read_text(encoding="utf-8"))["analysis"]["style_transfer_ability"]
    lpips = float(stats["content_lpips"])
    return [
        {
            "scope": "transfer",
            "family": "Reference",
            "label": "No-op transfer",
            "step_or_epoch": 0,
            "clip_style": float(stats["clip_style"]),
            "content_lpips": lpips,
            "one_minus_lpips": 1.0 - lpips,
            "train_min": 0.0,
            "note": "unchanged source, transfer-only off-diagonal pairs",
        }
    ]


def main() -> None:
    full = read_full_points()
    rows = full + collect_lancet_transfer(full) + collect_samam_transfer(full) + collect_noop_transfer()
    fields = ["scope", "family", "label", "step_or_epoch", "clip_style", "content_lpips", "one_minus_lpips", "train_min", "note"]
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(OUT_TABLE)


if __name__ == "__main__":
    main()
