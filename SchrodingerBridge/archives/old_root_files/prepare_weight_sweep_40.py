from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "S-add__K-1_C-0_W-20_Col-0" / "config.json"
OUT_ROOT = ROOT / "weight_sweep_40"
STYLE_ORDER = ["photo", "Hayao", "monet", "vangogh", "cezanne"]


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    description: str
    content_weights: list[float] | None
    target_weights: list[float] | None
    balance_target_styles_per_batch: bool = False


RECIPES: list[Recipe] = [
    Recipe("R00_balanced_default", "Original balanced target sampler from base config.", None, None, True),
    Recipe("R01_uniform_unbalanced", "Uniform content and target weights, unbalanced sampler.", [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]),
    Recipe("R02_prev_manual", "Previous manual recipe: more photo/Hayao content and Hayao/art target.", [1.35, 1.25, 0.85, 0.85, 0.85], [0.80, 1.35, 1.05, 1.05, 1.05]),
    Recipe("R03_hayao_strong", "Strong Hayao target pressure.", [1.30, 1.40, 0.85, 0.85, 0.85], [0.75, 1.70, 1.00, 1.00, 1.00]),
    Recipe("R04_cezanne_strong", "Repair Cezanne style drop by target emphasis.", [1.35, 1.05, 0.85, 0.85, 1.25], [0.75, 1.10, 1.00, 1.00, 1.60]),
    Recipe("R05_vangogh_strong", "Van Gogh target emphasis.", [1.35, 1.05, 0.85, 1.25, 0.85], [0.75, 1.10, 1.00, 1.60, 1.00]),
    Recipe("R06_monet_strong", "Monet target emphasis.", [1.35, 1.05, 1.25, 0.85, 0.85], [0.75, 1.10, 1.60, 1.00, 1.00]),
    Recipe("R07_art_balanced", "All art targets boosted equally, photo target suppressed.", [1.40, 1.00, 0.90, 0.90, 0.90], [0.70, 1.20, 1.20, 1.20, 1.20]),
    Recipe("R08_photo_content_high", "Content-heavy recipe: frequent photo content, moderate art target.", [1.80, 1.10, 0.75, 0.75, 0.75], [0.70, 1.20, 1.20, 1.20, 1.20]),
    Recipe("R09_photo_content_low", "Less photo as content; stress art-to-art content retention.", [0.90, 1.20, 1.10, 1.10, 1.10], [0.70, 1.25, 1.25, 1.25, 1.25]),
    Recipe("R10_no_photo_target", "Nearly remove photo target to focus transfer style.", [1.50, 1.10, 0.90, 0.90, 0.90], [0.30, 1.35, 1.35, 1.35, 1.35]),
    Recipe("R11_photo_target_some", "Keep more photo target for identity/content stability.", [1.50, 1.10, 0.90, 0.90, 0.90], [1.00, 1.20, 1.20, 1.20, 1.20]),
    Recipe("R12_hayao_cezanne", "Joint Hayao plus Cezanne target repair.", [1.35, 1.25, 0.80, 0.80, 1.10], [0.65, 1.45, 1.00, 1.00, 1.45]),
    Recipe("R13_monet_vangogh", "Joint Monet plus Van Gogh target pressure.", [1.35, 1.05, 1.10, 1.10, 0.80], [0.65, 1.10, 1.45, 1.45, 1.00]),
    Recipe("R14_soft_art", "Soft version of art target emphasis.", [1.25, 1.10, 0.95, 0.95, 0.95], [0.85, 1.15, 1.15, 1.15, 1.15]),
    Recipe("R15_hard_art", "Hard version of art target emphasis.", [1.60, 1.10, 0.75, 0.75, 0.75], [0.45, 1.45, 1.45, 1.45, 1.45]),
    Recipe("R16_photo_hayao_content_art_target", "Photo/Hayao content anchors with uniform art target pressure.", [1.55, 1.35, 0.75, 0.75, 0.75], [0.60, 1.25, 1.25, 1.25, 1.25]),
    Recipe("R17_art_content_art_target", "More art content and more art target; less photo dominance.", [0.85, 1.15, 1.15, 1.15, 1.15], [0.55, 1.30, 1.30, 1.30, 1.30]),
    Recipe("R18_cezanne_fix_prev", "Previous manual recipe plus extra Cezanne target.", [1.35, 1.25, 0.85, 0.85, 0.95], [0.75, 1.30, 1.05, 1.05, 1.45]),
    Recipe("R19_hayao_fix_prev", "Previous manual recipe plus extra Hayao target.", [1.35, 1.35, 0.80, 0.80, 0.80], [0.75, 1.60, 1.00, 1.00, 1.00]),
]


def _weight_str(values: list[float] | None) -> str:
    if values is None:
        return ""
    return ",".join(f"{v:g}" for v in values)


def _make_config(recipe: Recipe, k_value: float) -> tuple[str, dict]:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg = deepcopy(base)
    short = recipe.recipe_id.split("_", 1)[0].lower()
    name = f"K{int(k_value)}_{short}_{recipe.recipe_id.split('_', 1)[1]}"

    cfg.setdefault("bridge", {})
    cfg["bridge"]["w_kinetic"] = float(k_value)
    cfg["bridge"]["w_cycle"] = 0.0
    cfg["bridge"]["terminal_swd_weight"] = 20.0
    cfg["bridge"]["w_color"] = 0.0

    cfg.setdefault("training", {})
    cfg["training"]["num_epochs"] = 8
    cfg["training"]["save_interval"] = 1
    cfg["training"]["resume_checkpoint"] = ""
    cfg["training"]["full_eval_batch_size"] = 20

    cfg.setdefault("data", {})
    cfg["data"]["style_subdirs"] = STYLE_ORDER
    cfg["data"]["balance_target_styles_per_batch"] = recipe.balance_target_styles_per_batch
    if recipe.content_weights is None:
        cfg["data"].pop("content_style_sampling_weights", None)
    else:
        cfg["data"]["content_style_sampling_weights"] = recipe.content_weights
    if recipe.target_weights is None:
        cfg["data"].pop("target_style_sampling_weights", None)
    else:
        cfg["data"]["target_style_sampling_weights"] = recipe.target_weights

    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"]["save_dir"] = f"./weight_sweep_40/{name}"
    cfg["ablation"] = {
        "name": name,
        "axis": "weight_sweep_40",
        "recipe_id": recipe.recipe_id,
        "description": recipe.description,
        "style_order": STYLE_ORDER,
        "content_style_sampling_weights": recipe.content_weights,
        "target_style_sampling_weights": recipe.target_weights,
        "balance_target_styles_per_batch": recipe.balance_target_styles_per_batch,
        "score_note": "Primary score is clip_style * (1 - content_lpips); collection also reports weighted and normalized scores.",
    }
    return name, cfg


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    config_dir = OUT_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float | int | bool]] = []

    for k_value in (1.0, 2.0):
        for recipe in RECIPES:
            name, cfg = _make_config(recipe, k_value)
            config_path = config_dir / f"{name}.json"
            config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            rows.append(
                {
                    "experiment_id": name,
                    "k_value": k_value,
                    "recipe_id": recipe.recipe_id,
                    "description": recipe.description,
                    "balance_target_styles_per_batch": recipe.balance_target_styles_per_batch,
                    "content_weights": _weight_str(recipe.content_weights),
                    "target_weights": _weight_str(recipe.target_weights),
                    "config_path": str(config_path),
                    "run_dir": str(OUT_ROOT / name),
                }
            )

    manifest_csv = OUT_ROOT / "manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest_json = OUT_ROOT / "manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "base_config": str(BASE_CONFIG),
                "style_order": STYLE_ORDER,
                "num_experiments": len(rows),
                "recipes": [r.__dict__ for r in RECIPES],
                "experiments": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    md = [
        "# Weight Sweep 40",
        "",
        "Design: 20 category-sampling recipes x K={1,2}. All configs use the original K1 base config, `terminal_swd_weight=20`, `w_cycle=0`, `w_color=0`, `num_epochs=8`, `save_interval=1`.",
        "",
        "Primary scalar score: `clip_style * (1 - LPIPS)`. This is intentionally simple and interpretable: style strength weighted by content preservation. The runner also reports `score_weighted_65_35 = 0.65 * clip_style + 0.35 * (1 - LPIPS)` and min-max normalized variants.",
        "",
        "| experiment_id | K | recipe | content weights | target weights | note |",
        "|---|---:|---|---|---|---|",
    ]
    for row in rows:
        md.append(
            f"| {row['experiment_id']} | {row['k_value']} | {row['recipe_id']} | "
            f"{row['content_weights'] or 'balanced/default'} | {row['target_weights'] or 'balanced/default'} | {row['description']} |"
        )
    (OUT_ROOT / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} configs")
    print(manifest_csv)
    print(manifest_json)


if __name__ == "__main__":
    main()
