from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "exp/dwt_route_distinct5/config.json"
OUT_DIR = ROOT / "configs/semantic_swd_musiq"


def main() -> int:
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    common_bridge = {
        "spectral_w_ll": 0.3,
        "spectral_w_lh": 1.0,
        "spectral_w_hl": 1.0,
        "spectral_w_hh": 2.0,
        "w_channel_variance": 0.0,
        "w_pixel_color_match": 0.0,
        "single_step_swd_weight": 8.0,
        "single_step_edge_weight": 0.1,
        "terminal_swd_weight": 0.1,
    }
    common_training = {
        "num_epochs": 5,
        "save_interval": 5,
        "full_eval_defer_until_training_end": True,
        "full_eval_each_epoch": False,
        "full_eval_force_regen": True,
        "full_eval_save_generated_images": True,
        "full_eval_max_src_samples": None,
        "full_eval_max_ref_compare": None,
        "full_eval_max_ref_cache": None,
        "test_image_dir": "F:/wikiart_distinct5_samam_512_classview/test",
        "log_interval": 20,
    }
    common_full_eval = {
        "save_summary_grid": True,
        "max_ref_compare": 30,
        "max_ref_cache": 30,
    }
    variants = {
        "semantic_swd_global_clean5": {
            "bridge": {"swd_scale_mode": "global"},
            "notes": "Clean theory control: global SWD, weak LL, high-frequency spectral losses, no color/chvar auxiliaries.",
        },
        "semantic_swd_guided5": {
            "bridge": {
                "swd_scale_mode": "cross-attn-guided",
                "swd_guidance_source": "style_delta",
                "swd_guidance_floor": 0.25,
                "swd_guidance_power": 1.0,
                "swd_guidance_sample_size": 512,
            },
            "notes": "Semantic SWD: cross-attn style-delta guidance defines local SWD sampling mass.",
        },
        "semantic_swd_guided_cons5": {
            "bridge": {
                "swd_scale_mode": "cross-attn-guided",
                "swd_guidance_source": "style_delta",
                "swd_guidance_floor": 0.5,
                "swd_guidance_power": 0.5,
                "swd_guidance_sample_size": 512,
            },
            "notes": "Conservative semantic SWD: softer guidance to avoid over-localizing distribution matching.",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_config": str(BASE_PATH),
        "paper_table_refs": {
            "WEAVE_D5": {"CLIP-S": 0.7213, "LPIPS": 0.2868, "MUSIQ": 35.31},
            "SaMAM_D5": {"MUSIQ": 51.17},
            "SD-Turbo_D5": {"MUSIQ": 60.72},
            "Seedream_D5": {"MUSIQ": 69.51},
            "old_local_dwt_route": {
                "CLIP-S_allpairs": 0.7274521667162577,
                "LPIPS_allpairs": 0.4347057877466667,
                "MUSIQ": 41.10924326578776,
            },
        },
        "variants": {},
    }
    for name, spec in variants.items():
        cfg = deepcopy(base)
        cfg["model"]["dwt_route_train_prob"] = 0.8
        cfg["bridge"].update(common_bridge)
        cfg["bridge"].update(spec["bridge"])
        cfg["training"].update(common_training)
        cfg["full_eval"].update(common_full_eval)
        cfg["checkpoint"]["save_dir"] = f"./exp/{name}"
        cfg["ablation"] = {
            "name": name,
            "axis": "semantic_swd_musiq",
            "stage": "objective_test",
            "notes": spec["notes"],
        }
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        manifest["variants"][name] = str(path)
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    for name, path in manifest["variants"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
