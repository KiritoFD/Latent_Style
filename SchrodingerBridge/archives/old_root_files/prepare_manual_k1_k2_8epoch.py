from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "S-add__K-1_C-0_W-20_Col-0" / "config.json"
OUT_ROOT = ROOT / "manual_k1_k2_8epoch"


def write_config(name: str, w_kinetic: float, *, weighted: bool = False) -> Path:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg = deepcopy(base)
    cfg.setdefault("bridge", {})
    cfg["bridge"]["w_kinetic"] = float(w_kinetic)
    cfg["bridge"]["w_cycle"] = 0.0
    cfg["bridge"]["terminal_swd_weight"] = 20.0
    cfg["bridge"]["w_color"] = 0.0

    cfg.setdefault("training", {})
    cfg["training"]["num_epochs"] = 8
    cfg["training"]["save_interval"] = 1
    cfg["training"]["resume_checkpoint"] = ""

    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"]["save_dir"] = f"./manual_k1_k2_8epoch/{name}"

    if weighted:
        # style_subdirs = [photo, Hayao, monet, vangogh, cezanne].
        # The current weak spots are photo->art strength and Hayao transfers.
        # Keep photo frequent as content, boost Hayao and art targets.
        cfg.setdefault("data", {})
        cfg["data"]["balance_target_styles_per_batch"] = False
        cfg["data"]["content_style_sampling_weights"] = [1.35, 1.25, 0.85, 0.85, 0.85]
        cfg["data"]["target_style_sampling_weights"] = [0.80, 1.35, 1.05, 1.05, 1.05]

    cfg["ablation"] = {
        "name": name,
        "axis": "manual_k1_k2_8epoch",
        "notes": f"Manual clean 8-epoch run from K1 config; w_kinetic={w_kinetic}, terminal_swd_weight=20, w_cycle=0, w_color=0, weighted={weighted}.",
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    configs = [
        write_config("K1_style_8ep_repro", 1.0),
        write_config("K2_content_8ep_repro", 2.0),
        write_config("K1_manual_weighted_8ep", 1.0, weighted=True),
        write_config("K2_manual_weighted_8ep", 2.0, weighted=True),
    ]
    manifest = {"base_config": str(BASE_CONFIG), "configs": [str(p) for p in configs]}
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in configs:
        print(path)


if __name__ == "__main__":
    main()
