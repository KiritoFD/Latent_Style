from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ABLATIONS = [
    (
        "00_full_ot_swd_region",
        {
            "model": {
                "cross_attn_dwt_route": True,
                "dwt_route_train_prob": 0.8,
                "endpoint_adain_scale": 1.0,
            },
            "bridge": {
                "single_step_swd_weight": 12.0,
                "swd_scale_mode": "cross-attn-guided",
                "swd_guidance_sample_size": 512,
                "swd_region_sample_size": 256,
                "swd_semantic_mode": "region",
                "swd_semantic_regions": 8,
                "swd_semantic_blend": 0.7,
                "swd_semantic_kmeans_iters": 3,
                "spectral_w_ll": 0.3,
            },
            "full_eval": {"hf_soft_threshold": 0.08},
        },
    ),
    (
        "01_no_swd",
        {
            "model": {
                "cross_attn_dwt_route": True,
                "dwt_route_train_prob": 0.8,
                "endpoint_adain_scale": 1.0,
            },
            "bridge": {
                "single_step_swd_weight": 0.0,
                "swd_scale_mode": "global",
                "swd_semantic_mode": "off",
                "spectral_w_ll": 0.3,
            },
            "full_eval": {"hf_soft_threshold": 0.08},
        },
    ),
    (
        "02_global_swd_no_region",
        {
            "model": {
                "cross_attn_dwt_route": True,
                "dwt_route_train_prob": 0.8,
                "endpoint_adain_scale": 1.0,
            },
            "bridge": {
                "single_step_swd_weight": 12.0,
                "swd_scale_mode": "cross-attn-guided",
                "swd_guidance_sample_size": 512,
                "swd_semantic_mode": "off",
                "spectral_w_ll": 0.3,
            },
            "full_eval": {"hf_soft_threshold": 0.08},
        },
    ),
    (
        "03_no_dwt_route",
        {
            "model": {
                "cross_attn_dwt_route": False,
                "dwt_route_train_prob": 0.0,
                "endpoint_adain_scale": 1.0,
            },
            "bridge": {
                "single_step_swd_weight": 12.0,
                "swd_scale_mode": "cross-attn-guided",
                "swd_guidance_sample_size": 512,
                "swd_region_sample_size": 256,
                "swd_semantic_mode": "region",
                "swd_semantic_regions": 8,
                "swd_semantic_blend": 0.7,
                "swd_semantic_kmeans_iters": 3,
                "spectral_w_ll": 0.3,
            },
            "full_eval": {"hf_soft_threshold": 0.08},
        },
    ),
    (
        "04_no_endpoint_wct",
        {
            "model": {
                "cross_attn_dwt_route": True,
                "dwt_route_train_prob": 0.8,
                "endpoint_adain_scale": 0.0,
            },
            "bridge": {
                "single_step_swd_weight": 12.0,
                "swd_scale_mode": "cross-attn-guided",
                "swd_guidance_sample_size": 512,
                "swd_region_sample_size": 256,
                "swd_semantic_mode": "region",
                "swd_semantic_regions": 8,
                "swd_semantic_blend": 0.7,
                "swd_semantic_kmeans_iters": 3,
                "spectral_w_ll": 0.3,
            },
            "full_eval": {"hf_soft_threshold": 0.08},
        },
    ),
    (
        "05_no_ll_deweight",
        {
            "model": {
                "cross_attn_dwt_route": True,
                "dwt_route_train_prob": 0.8,
                "endpoint_adain_scale": 1.0,
            },
            "bridge": {
                "single_step_swd_weight": 12.0,
                "swd_scale_mode": "cross-attn-guided",
                "swd_guidance_sample_size": 512,
                "swd_region_sample_size": 256,
                "swd_semantic_mode": "region",
                "swd_semantic_regions": 8,
                "swd_semantic_blend": 0.7,
                "swd_semantic_kmeans_iters": 3,
                "spectral_w_ll": 1.0,
            },
            "full_eval": {"hf_soft_threshold": 0.08},
        },
    ),
    (
        "06_no_eota_eval",
        {
            "model": {
                "cross_attn_dwt_route": True,
                "dwt_route_train_prob": 0.8,
                "endpoint_adain_scale": 1.0,
            },
            "bridge": {
                "single_step_swd_weight": 12.0,
                "swd_scale_mode": "cross-attn-guided",
                "swd_guidance_sample_size": 512,
                "swd_region_sample_size": 256,
                "swd_semantic_mode": "region",
                "swd_semantic_regions": 8,
                "swd_semantic_blend": 0.7,
                "swd_semantic_kmeans_iters": 3,
                "spectral_w_ll": 0.3,
            },
            "full_eval": {"hf_soft_threshold": 0.0},
        },
    ),
    (
        "07_all_off_fm_only",
        {
            "model": {
                "cross_attn_dwt_route": False,
                "dwt_route_train_prob": 0.0,
                "endpoint_adain_scale": 0.0,
            },
            "bridge": {
                "single_step_swd_weight": 0.0,
                "swd_scale_mode": "global",
                "swd_semantic_mode": "off",
                "spectral_w_ll": 1.0,
            },
            "full_eval": {"hf_soft_threshold": 0.0},
        },
    ),
    (
        "08_region_only_no_xattn",
        {
            "model": {
                "cross_attn_dwt_route": False,
                "dwt_route_train_prob": 0.0,
                "endpoint_adain_scale": 1.0,
            },
            "bridge": {
                "single_step_swd_weight": 12.0,
                "swd_scale_mode": "global",
                "swd_guidance_sample_size": 512,
                "swd_region_sample_size": 256,
                "swd_semantic_mode": "region",
                "swd_semantic_regions": 8,
                "swd_semantic_blend": 0.7,
                "swd_semantic_kmeans_iters": 3,
                "spectral_w_ll": 0.3,
            },
            "full_eval": {"hf_soft_threshold": 0.08},
        },
    ),
    (
        "09_overstyle_swd24_region16",
        {
            "model": {
                "cross_attn_dwt_route": True,
                "dwt_route_train_prob": 1.0,
                "endpoint_adain_scale": 1.5,
            },
            "bridge": {
                "single_step_swd_weight": 24.0,
                "swd_scale_mode": "cross-attn-guided",
                "swd_guidance_sample_size": 512,
                "swd_region_sample_size": 128,
                "swd_semantic_mode": "region",
                "swd_semantic_regions": 16,
                "swd_semantic_blend": 1.0,
                "swd_semantic_kmeans_iters": 3,
                "spectral_w_ll": 0.0,
            },
            "full_eval": {"hf_soft_threshold": 0.12},
        },
    ),
]


def deep_update(dst: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strong SWD/OT ablation configs for remote runs.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("exp/semantic_swd_ref_guided_cons5/config.json"),
        help="Known-good base config, relative to SchrodingerBridge unless absolute.",
    )
    parser.add_argument("--out-root", type=Path, default=Path("exp/remote_swd_ablation"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-src-samples", type=int, default=48)
    parser.add_argument("--max-ref-compare", type=int, default=32)
    parser.add_argument("--remote-data-root", default="I:/wikiart_distinct5_samam_512_latents_ema/train")
    parser.add_argument("--remote-test-root", default="I:/wikiart_distinct5_samam_512_classview/test")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    base_path = args.base if args.base.is_absolute() else repo / args.base
    out_root = args.out_root if args.out_root.is_absolute() else repo / args.out_root
    base = json.loads(base_path.read_text(encoding="utf-8"))
    written: list[Path] = []

    for name, patch in ABLATIONS:
        cfg = copy.deepcopy(base)
        cfg.setdefault("checkpoint", {})["save_dir"] = f"./exp/remote_swd_ablation/{name}"
        cfg.setdefault("data", {}).update(
            {
                "data_root": str(args.remote_data_root),
                "latent_cache_dir": str(Path(args.remote_data_root) / ".latent_cache" / "packed"),
                "latent_cache_mode": "packed",
                "test_image_dir": str(args.remote_test_root),
            }
        )
        cfg.setdefault("training", {}).update(
            {
                "num_epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "use_tqdm": False,
                "save_interval": int(args.epochs),
                "full_eval_each_epoch": False,
                "full_eval_defer_until_training_end": True,
                "full_eval_force_regen": True,
                "full_eval_only_lpips_clip_style": True,
                "full_eval_max_src_samples": int(args.max_src_samples),
                "full_eval_max_ref_compare": int(args.max_ref_compare),
                "full_eval_batch_size": 4,
                "gpu_monitor_interval_sec": 5.0,
            }
        )
        cfg.setdefault("full_eval", {}).update(
            {
                "only_lpips_clip_style": True,
                "max_src_samples": int(args.max_src_samples),
                "max_ref_compare": int(args.max_ref_compare),
                "batch_size": 4,
            }
        )
        deep_update(cfg, patch)

        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "config.json"
        out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(out_path)

    manifest = out_root / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in written) + "\n", encoding="utf-8")
    print(f"Wrote {len(written)} configs under {out_root}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
