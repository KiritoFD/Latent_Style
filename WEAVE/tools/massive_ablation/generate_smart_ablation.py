#!/usr/bin/env python3
"""Smart Grouped Ablation Sweep: ~180 experiments.
Uses fine-grained axis sweeping around a solid baseline instead of naive full Cartesian product.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from itertools import product
from pathlib import Path
from typing import Any

def _load_base(base_path: str | None = None) -> dict:
    if base_path is None:
        base_path = Path(__file__).with_name("base_focused.json")
    with open(base_path, "r", encoding="utf-8") as f:
        return json.load(f)

BASE = _load_base()

def _set_path(config: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    obj = config
    for part in parts[:-1]:
        obj = obj.setdefault(part, {})
    obj[parts[-1]] = value

def _get_path(config: dict, path: str) -> Any:
    parts = path.split(".")
    obj = config
    for part in parts:
        obj = obj[part]
    return obj

def _clone(base: dict) -> dict:
    return json.loads(json.dumps(base))

def _make_name(overrides: list[tuple[str, str, Any]]) -> str:
    parts = []
    for display, _, val in overrides:
        v = str(val).replace(".", "p").replace(" ", "")
        if isinstance(val, bool):
            v = "T" if val else "F"
        parts.append(f"{display}_{v}")
    return "abl_" + "_".join(parts)

def _build_experiment(overrides: list[tuple[str, str, Any]], group: str) -> tuple[str, dict, list]:
    cfg = _clone(BASE)
    for display, path, val in overrides:
        _set_path(cfg, path, val)
    name = _make_name(overrides)
    cfg["checkpoint"] = {"save_dir": f"./exp/620_smart_ablation/{name}"}
    cfg["ablation"] = {"name": name, "group": group, "stage": "smart_ablation"}
    return name, cfg, overrides

def generate_smart_groups() -> list:
    experiments = []
    
    # ==========================================
    # Group A: Texture SWD vs Target Projection (30 runs)
    # ==========================================
    swd_vals = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 80.0, 120.0, 200.0]
    proj_vals = ["legacy", "source_low_target_high", "pure_vertical_flow"]
    for swd, proj in product(swd_vals, proj_vals):
        experiments.append(_build_experiment([
            ("swd", "bridge.single_step_swd_weight", swd),
            ("proj", "bridge.training_target_projection_mode", proj)
        ], "A_Texture_vs_Projection"))

    # ==========================================
    # Group B: ODE Flow vs SDE Noise (48 runs)
    # ==========================================
    flow_vals = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
    sigma_vals = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
    for flow, sig in product(flow_vals, sigma_vals):
        experiments.append(_build_experiment([
            ("flow", "bridge.w_flow", flow),
            ("sig", "bridge.bridge_sigma", sig)
        ], "B_Flow_vs_Sigma"))

    # ==========================================
    # Group C: Model Capacity Map (20 runs)
    # ==========================================
    dim_vals = [32, 48, 64, 96, 128]
    blk_vals = [2, 4, 6, 8]
    for dim, blk in product(dim_vals, blk_vals):
        experiments.append(_build_experiment([
            ("dim", "model.base_dim", dim),
            ("blk", "model.num_res_blocks", blk)
        ], "C_Capacity"))

    # ==========================================
    # Group D: Conditioning Routing (24 runs)
    # ==========================================
    cond_vals = ["latent", "target_dino_patches"]
    shortcut_vals = [0.0, 0.2, 0.5, 1.0]
    attn_vals = ["softmax", "sparsemax", "gated"]
    for cond, short, attn in product(cond_vals, shortcut_vals, attn_vals):
        experiments.append(_build_experiment([
            ("cond", "model.style_condition_source", cond),
            ("short", "model.style_shortcut_alpha", short),
            ("attn", "model.style_attn_mode", attn)
        ], "D_Routing"))

    # ==========================================
    # Group E: Endpoint & High-Freq Amplification (16 runs after pruning)
    # ==========================================
    # Logical Pruning: If mode is 'velocity', kernel and high_scale do nothing.
    for ep_mode in ["velocity", "endpoint_lowhigh"]:
        if ep_mode == "velocity":
            experiments.append(_build_experiment([("ep", "model.endpoint_head_mode", ep_mode)], "E_Endpoint"))
        else:
            kernel_vals = [3, 5, 9, 15]
            high_scale_vals = [0.5, 1.0, 1.5, 2.0]
            for k, hs in product(kernel_vals, high_scale_vals):
                experiments.append(_build_experiment([
                    ("ep", "model.endpoint_head_mode", ep_mode),
                    ("k", "model.endpoint_lowpass_kernel", k),
                    ("hs", "model.endpoint_high_scale", hs)
                ], "E_Endpoint"))

    # ==========================================
    # Group F: Content Anchoring vs Projection Anchoring (20 runs)
    # ==========================================
    content_anchors = [0.0, 0.1, 0.5, 1.0, 5.0]
    proj_anchors = [0.0, 0.5, 0.8, 1.0]
    for ca, pa in product(content_anchors, proj_anchors):
        experiments.append(_build_experiment([
            ("ca", "bridge.w_content_lowpass_anchor", ca),
            ("pa", "bridge.training_target_projection_low_anchor", pa)
        ], "F_Anchoring"))

    # Deduplicate
    seen = set()
    final = []
    for name, cfg, overrides in experiments:
        if name in seen:
            continue
        seen.add(name)
        final.append((name, cfg, overrides))
    
    # Sort logically by group
    final.sort(key=lambda x: x[1]["ablation"]["group"])
    return final

def _finalize_config(cfg: dict, src_root: str, exp_root: str) -> dict:
    cfg = _clone(cfg)
    src_root_posix = Path(src_root).as_posix()
    cfg["data"]["data_root"] = src_root_posix
    cfg["data"]["latent_cache_dir"] = f"{src_root_posix}/.latent_cache/packed"
    cfg["data"]["pairing_cache_path"] = f"{src_root_posix}/.latent_cache/dino_pairing_top8.pt"
    cfg["checkpoint"]["save_dir"] = Path(exp_root).as_posix() + f"/{cfg['ablation']['name']}"
    cfg["data"]["dino_cache_required"] = (_get_path(cfg, "model.style_condition_source") == "target_dino_patches")
    return cfg

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(Path(__file__).parent), help="Output directory")
    parser.add_argument("--src-root", default="/mnt/i/wikiart_distinct5_samam_512_latents_ema/train")
    parser.add_argument("--exp-root", default="./exp/620_smart_ablation")
    parser.add_argument("--base", default=None, help="Path to base_focused.json")
    args = parser.parse_args()

    global BASE
    BASE = _load_base(args.base)

    base_outdir = Path(args.outdir)
    cfg_dir = base_outdir / "configs_smart"
    if cfg_dir.exists():
        shutil.rmtree(cfg_dir, ignore_errors=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    experiments = generate_smart_groups()
    print(f"Total smart experiments: {len(experiments)}")

    # Write configs
    for name, cfg, overrides in experiments:
        cfg = _finalize_config(cfg, args.src_root, args.exp_root)
        with open(cfg_dir / f"{name}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    # Write Matrix
    with open(base_outdir / "matrix_smart.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["idx", "name", "group", "overrides"])
        for i, (name, cfg, overrides) in enumerate(experiments):
            ov_str = "; ".join(f"{n}={v}" for n, _, v in overrides)
            writer.writerow([i, name, cfg["ablation"]["group"], ov_str])

    # Write Launcher
    config_base = Path("tools/massive_ablation/configs_smart").as_posix()
    exp_base = Path(args.exp_root).as_posix()
    launch_path = base_outdir / "run_smart_ablation.sh"
    
    with open(launch_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Smart ablation: {len(experiments)} experiments.\n")
        f.write("set -euo pipefail\n\n")
        f.write('ROOT="$(cd "$(dirname "$0")/../.." && pwd)"\n')
        f.write(f'CONFIG_BASE="$ROOT/{config_base}"\n')
        f.write(f'EXP_BASE="$ROOT/{exp_base}"\n\n')
        f.write('cd "$ROOT"\n')
        f.write('if [ ! -d "$CONFIG_BASE" ]; then\n')
        f.write('  echo "Generating smart configs..."\n')
        f.write('  python "$ROOT/tools/massive_ablation/generate_smart_ablation.py"\n')
        f.write('fi\n\n')
        f.write(f'TOTAL={len(experiments)}\n')
        f.write("COUNT=0\nFAILED=0\nFAILED_LIST=\"\"\n\n")
        f.write("run_one() {\n")
        f.write("  local NAME=$1\n")
        f.write('  local OUTDIR="$EXP_BASE/$NAME"\n')
        f.write("  mkdir -p \"$OUTDIR\"\n")
        f.write('  cp "$CONFIG_BASE/${NAME}.json" "$OUTDIR/config.json"\n')
        f.write('  echo "\n===================================================================" \n')
        f.write('  echo "[$COUNT/$TOTAL] smart: $NAME"\n')
        f.write('  echo "==================================================================="\n')
        f.write('  if [ -f "$OUTDIR/full_eval/curve_summary.json" ]; then\n')
        f.write('    echo "SKIP $NAME (already evaluated)"\n')
        f.write('    COUNT=$((COUNT+1)); return\n')
        f.write('  fi\n')
        f.write('  if python run.py --config "$OUTDIR/config.json" 2>&1 | tee "$OUTDIR/smart.log"; then\n')
        f.write('    local RC=0\n  else\n    local RC=$?\n')
        f.write('    echo "  FAILED $NAME (rc=$RC)"\n')
        f.write('    FAILED=$((FAILED+1))\n    FAILED_LIST="$FAILED_LIST $NAME"\n  fi\n')
        f.write('  COUNT=$((COUNT+1))\n}\n\n')
        
        for name, _, _ in experiments:
            f.write(f'run_one "{name}"\n')
        
        f.write("\necho \"SMART DONE. Total=$TOTAL OK=$((TOTAL-FAILED)) Failed=$FAILED\"\n")
    os.chmod(launch_path, 0o755)

    print("\nGenerated smart scripts successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
