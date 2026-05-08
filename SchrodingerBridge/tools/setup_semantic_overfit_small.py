from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
BASE_CONFIG_PATH = REPO_ROOT / "config.json"
LATENT_ROOT = PROJECT_ROOT / "latent-256"
IMAGE_ROOT = PROJECT_ROOT / "style_data" / "overfit50"
OUT_ROOT = REPO_ROOT / "experiments" / "semantic_overfit_small"
LATENT_OUT = OUT_ROOT / "latents"
IMAGE_OUT = OUT_ROOT / "images"


def _first_file(directory: Path, patterns: tuple[str, ...]) -> Path:
    for pattern in patterns:
        files = sorted(directory.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(f"No files matching {patterns} in {directory}")


def _copy_one(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def main() -> None:
    base_cfg = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))

    photo_latent = _first_file(LATENT_ROOT / "photo", ("*.pt", "*.npy"))
    monet_latent = _first_file(LATENT_ROOT / "monet", ("*.pt", "*.npy"))
    photo_image = _first_file(IMAGE_ROOT / "photo", ("*.jpg", "*.png", "*.jpeg"))
    monet_image = _first_file(IMAGE_ROOT / "monet", ("*.jpg", "*.png", "*.jpeg"))

    copied_photo_latent = _copy_one(photo_latent, LATENT_OUT / "photo")
    copied_monet_latent = _copy_one(monet_latent, LATENT_OUT / "monet")
    copied_photo_image = _copy_one(photo_image, IMAGE_OUT / "photo")
    copied_monet_image = _copy_one(monet_image, IMAGE_OUT / "monet")

    experiments = {
        "A_baseline": {
            "w_kinetic": 1.0,
            "w_low_freq": 1.0,
            "terminal_swd_weight": 0.1,
            "w_cycle": 1.0,
        },
        "B_no_kinetic": {
            "w_kinetic": 0.0,
            "w_low_freq": 1.0,
            "terminal_swd_weight": 0.5,
            "w_cycle": 1.0,
        },
        "C_no_low_freq": {
            "w_kinetic": 1.0,
            "w_low_freq": 0.0,
            "terminal_swd_weight": 0.1,
            "w_cycle": 1.0,
        },
    }

    manifest: dict[str, object] = {
        "sandbox": {
            "photo_latent": str(copied_photo_latent),
            "monet_latent": str(copied_monet_latent),
            "photo_image": str(copied_photo_image),
            "monet_image": str(copied_monet_image),
        },
        "experiments": [],
    }

    configs_dir = OUT_ROOT / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    for name, bridge_overrides in experiments.items():
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("model", {})
        cfg.setdefault("bridge", {})
        cfg.setdefault("training", {})
        cfg.setdefault("data", {})
        cfg.setdefault("checkpoint", {})

        cfg["model"]["num_styles"] = 2

        cfg["data"]["data_root"] = "./experiments/semantic_overfit_small/latents"
        cfg["data"]["style_subdirs"] = ["photo", "monet"]
        cfg["data"]["allow_hflip"] = False
        cfg["data"]["identity_ratio"] = 0.0
        cfg["data"]["balance_target_styles_per_batch"] = False
        cfg["data"]["virtual_length_multiplier"] = 256
        cfg["data"]["preload_to_gpu"] = False

        cfg["training"]["batch_size"] = 2
        cfg["training"]["num_workers"] = 0
        cfg["training"]["persistent_workers"] = False
        cfg["training"]["shuffle"] = False
        cfg["training"]["num_epochs"] = 3
        cfg["training"]["save_interval"] = 1
        cfg["training"]["log_interval"] = 10
        cfg["training"]["full_eval_batch_size"] = 1
        cfg["training"]["test_image_dir"] = "./experiments/semantic_overfit_small/images"

        cfg["checkpoint"]["save_dir"] = f"./experiments/semantic_overfit_small/{name}"

        cfg["bridge"]["objective_mode"] = "omf"
        cfg["bridge"]["loss_type"] = "omf"
        cfg["bridge"]["w_flow"] = 0.0
        cfg["bridge"]["w_color"] = 0.0
        cfg["bridge"]["w_repulsive"] = 0.0
        cfg["bridge"]["w_nce"] = 0.0
        cfg["bridge"]["semantic_swd_num_projections"] = 16
        cfg["bridge"]["low_freq_kernel_size"] = 7
        for key, value in bridge_overrides.items():
            cfg["bridge"][key] = value

        out_path = configs_dir / f"{name}.json"
        out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        manifest["experiments"].append(
            {
                "name": name,
                "config": str(out_path),
                "checkpoint_dir": cfg["checkpoint"]["save_dir"],
                "expected_iterations_per_epoch": 256,
                "expected_total_iterations": 768,
                "bridge": bridge_overrides,
            }
        )

    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created sandbox under {OUT_ROOT}")


if __name__ == "__main__":
    main()
