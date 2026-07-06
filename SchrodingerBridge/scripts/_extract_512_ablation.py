#!/usr/bin/env python3
"""Extract 512 ablation metrics from existing FCSB phase4 experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path("g:/GitHub/Latent_Style/SchrodingerBridge/exp/FCSB/phase4")

EXPERIMENTS = {
    # name -> directory name
    "full_630_phase4i2b_sota_heun_5ep": "630_phase4i2b_sota_heun_5ep",
    "no_dwt_630_phase4a2_adain_0": "630_phase4a2_adain_0",
    "no_endpoint_wct_630_phase4a2_w_ll_0": "630_phase4a2_w_ll_0",  # may not be no-endpoint-wct
    "no_extrap_630_phase4a2_extrap_0": "630_phase4a2_extrap_0",
    "lock_ll_630_phase4g1a_lock_ll": "630_phase4g1a_lock_ll",
    "per_step_adain_630_phase4g2_per_subband": "630_phase4g2_per_subband",
    "endpoint_ll_inject_630_phase4i10b_ept_t01": "630_phase4i10b_ept_t01",
    "routing_p0_630_phase4j1_dwt_route": "630_phase4j1_dwt_route",
    "routing_p05_630_phase4h5e_sota_mask25": "630_phase4h5e_sota_mask25",
    "routing_p1_630_phase4h5f_sota_mask75": "630_phase4h5f_sota_mask75",
    "w_ll_05_630_phase4h2i_per_subband_a07_w_ll_05": "630_phase4h2i_per_subband_a07_w_ll_05",
    "w_ll_10_630_phase4j2_wct_aligned": "630_phase4j2_wct_aligned",
    "no_endpoint_wct_630_phase4d_lvl2": "630_phase4d_lvl2",
    "no_endpoint_wct_630_phase4f_lvl3": "630_phase4f_lvl3",
}


def extract(summary_path: Path) -> dict | None:
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    analysis = summary.get("analysis", {})
    transfer = analysis.get("style_transfer_ability", {})
    allpairs = analysis.get("all_pairs_overview", {})
    idt = analysis.get("identity_reconstruction", {})
    return {
        "t_clip_s": transfer.get("clip_style"),
        "t_lpips": transfer.get("content_lpips"),
        "ap_clip_s": allpairs.get("clip_style"),
        "ap_lpips": allpairs.get("content_lpips"),
        "idt_clip_s": idt.get("clip_style"),
        "idt_lpips": idt.get("content_lpips"),
    }


def main() -> int:
    rows = []
    for name, dirname in EXPERIMENTS.items():
        exp_dir = BASE / dirname
        # prefer epoch_0005, then epoch_0003
        for epoch in ["epoch_0005", "epoch_0003", "epoch_0004", "epoch_0002", "epoch_0001"]:
            summary_path = exp_dir / "full_eval" / epoch / "summary.json"
            metrics = extract(summary_path)
            if metrics:
                rows.append((name, dirname, epoch, metrics))
                break
        else:
            rows.append((name, dirname, "NOT_FOUND", None))

    print(f"{'name':<45} {'epoch':<12} {'tCLIP-S':>8} {'tLPIPS':>8} {'apCLIP-S':>8} {'apLPIPS':>8}")
    print("-" * 100)
    for name, dirname, epoch, metrics in rows:
        if metrics:
            print(f"{name:<45} {epoch:<12} {metrics['t_clip_s']:>8.4f} {metrics['t_lpips']:>8.4f} {metrics['ap_clip_s']:>8.4f} {metrics['ap_lpips']:>8.4f}")
        else:
            print(f"{name:<45} {epoch:<12} NOT_FOUND")
    return 0


if __name__ == "__main__":
    sys.exit(main())
