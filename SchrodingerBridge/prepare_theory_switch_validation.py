from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "S-add__K-1_C-0_W-20_Col-0" / "config.json"
OUT_ROOT = ROOT / "theory_switch_validation"


@dataclass(frozen=True)
class Variant:
    name: str
    note: str
    model_updates: dict[str, Any]
    bridge_updates: dict[str, Any]


VARIANTS = [
    Variant(
        "T0_k2_baseline",
        "K2 baseline from the same base config; no new switch enabled.",
        {},
        {},
    ),
    Variant(
        "T1_sinkhorn_routing",
        "Semantic attention uses Sinkhorn-style doubly normalized routing.",
        {"semantic_attn_routing_mode": "sinkhorn", "semantic_sinkhorn_iters": 3},
        {},
    ),
    Variant(
        "T2_entropy_gate_2p5",
        "Kinetic penalty gated by semantic attention entropy, moderate strength.",
        {},
        {"kinetic_entropy_gate_weight": 2.5},
    ),
    Variant(
        "T3_entropy_gate_5p0",
        "Kinetic penalty gated by semantic attention entropy, strong strength.",
        {},
        {"kinetic_entropy_gate_weight": 5.0},
    ),
    Variant(
        "T4_sinkhorn_entropy",
        "Sinkhorn routing plus moderate entropy-gated kinetic penalty.",
        {"semantic_attn_routing_mode": "sinkhorn", "semantic_sinkhorn_iters": 3},
        {"kinetic_entropy_gate_weight": 2.5},
    ),
    Variant(
        "T5_color_soft_w2",
        "Mild contextual color loss with regular softmax transport.",
        {},
        {"w_color": 2.0, "color_transport_mode": "softmax"},
    ),
    Variant(
        "T6_color_gumbel_w2",
        "Mild contextual color loss with hard Gumbel transport.",
        {},
        {"w_color": 2.0, "color_transport_mode": "gumbel_hard", "color_gumbel_tau": 1.0},
    ),
    Variant(
        "T7_all_switches_mild",
        "Combined mild package: Sinkhorn routing, entropy gate, and Gumbel color transport.",
        {"semantic_attn_routing_mode": "sinkhorn", "semantic_sinkhorn_iters": 3},
        {"kinetic_entropy_gate_weight": 2.5, "w_color": 2.0, "color_transport_mode": "gumbel_hard", "color_gumbel_tau": 1.0},
    ),
]


def make_config(variant: Variant) -> dict[str, Any]:
    cfg = deepcopy(json.loads(BASE_CONFIG.read_text(encoding="utf-8")))
    cfg.setdefault("bridge", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("checkpoint", {})

    # Match the strongest short-run setting found in the weight sweep family.
    cfg["bridge"]["w_kinetic"] = 2.0
    cfg["bridge"]["w_cycle"] = 0.0
    cfg["bridge"]["terminal_swd_weight"] = 20.0
    cfg["bridge"]["w_color"] = 0.0
    cfg["bridge"].update(variant.bridge_updates)
    cfg["model"].update(variant.model_updates)

    cfg["training"]["num_epochs"] = 3
    cfg["training"]["save_interval"] = 1
    cfg["training"]["resume_checkpoint"] = ""
    cfg["training"]["full_eval_batch_size"] = 20

    cfg["checkpoint"]["save_dir"] = f"./theory_switch_validation/{variant.name}"
    cfg["ablation"] = {
        "name": variant.name,
        "axis": "theory_switch_validation",
        "note": variant.note,
        "model_updates": variant.model_updates,
        "bridge_updates": variant.bridge_updates,
        "protocol": "3 epochs, evaluate every epoch on strict 750 protocol.",
    }
    return cfg


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    config_dir = OUT_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        cfg = make_config(variant)
        config_path = config_dir / f"{variant.name}.json"
        config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(
            {
                "experiment_id": variant.name,
                "note": variant.note,
                "config_path": str(config_path),
                "run_dir": str(OUT_ROOT / variant.name),
                "model_updates": json.dumps(variant.model_updates, ensure_ascii=False),
                "bridge_updates": json.dumps(variant.bridge_updates, ensure_ascii=False),
            }
        )
    with (OUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps({"base_config": str(BASE_CONFIG), "variants": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} validation configs")
    print(OUT_ROOT / "manifest.csv")


if __name__ == "__main__":
    main()
