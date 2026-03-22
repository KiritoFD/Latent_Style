import copy
import json
from pathlib import Path


def load_base_config() -> dict:
    base_path = Path(__file__).resolve().parent / "config_decoder-D-sweetspot.json"
    with open(base_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cleanup_old_outputs(out_dir: Path) -> None:
    old_patterns = [
        "config_color_0*_*.json",
        "config_color_ablation_exp*.json",
        "config_weight_exp*.json",
        "config_style_oa_*.json",
        "config_tmp_batch*.json",
        "run_color_r16_e60.bat",
        "run_color_ablation_anchor_4.bat",
        "anchor4.bat",
        "weight.bat",
        "style_oa.bat",
    ]
    removed = 0
    for pattern in old_patterns:
        for p in out_dir.glob(pattern):
            if p.is_file():
                p.unlink(missing_ok=True)
                removed += 1
    if removed > 0:
        print(f"cleaned old files: {removed}")


def generate_style_oa8() -> None:
    base = load_base_config()
    out_dir = Path(__file__).resolve().parent
    _cleanup_old_outputs(out_dir)

    base.setdefault("model", {})
    base["model"]["ada_mix_rank"] = 16

    base.setdefault("training", {})
    # Remove no-op/deprecated training keys inherited from legacy base config.
    # - profile_loss_vram: not consumed in trainer/run path
    # - empty_cache_interval: forcibly disabled in trainer for allocator stability
    # - loss_timing_interval: forcibly disabled in trainer for throughput
    for k in ("profile_loss_vram", "empty_cache_interval", "loss_timing_interval"):
        base["training"].pop(k, None)
    base["training"]["num_epochs"] = 120
    base["training"]["full_eval_interval"] = 60
    base["training"]["save_interval"] = 60
    base["training"]["full_eval_on_last_epoch"] = True
    base["training"]["batch_size"] = 320
    base["training"]["learning_rate"] = 2e-4
    base["training"]["min_learning_rate"] = 2e-5
    base["training"]["grad_clip_norm"] = 1.0
    base["training"]["warmup_ratio"] = 0.08
    base["training"]["warmup_start_factor"] = 0.0

    base.setdefault("loss", {})
    for k in (
        "w_latent_color",
        "w_delta_tv",
        "latent_color_mode",
        "latent_color_pool",
        "latent_color_blur",
        "latent_color_w_mean",
        "latent_color_w_std",
        "latent_color_w_cov",
        "use_nce",
        "w_nce",
        "nce_tau",
        "nce_num_patches",
        "nce_pool",
        "nce_mode",
    ):
        base["loss"].pop(k, None)
    base["loss"]["w_swd"] = 60.0
    base["loss"]["w_identity"] = 1.5
    base["loss"]["w_color"] = 2.0
    base["loss"]["color_eps"] = 1e-6
    base["loss"]["color_legacy_pool"] = 4
    base["loss"]["color_mode"] = "latent_decoupled_adain"

    base.setdefault("data", {})
    # image_root is not used in the train/run path for latent dataset training.
    base["data"].pop("image_root", None)

    experiments = [
        ("style_oa_1_lr2e4_wc2_swd60_id15_e120", {"learning_rate": 2e-4, "w_color": 2.0, "w_swd": 60.0, "w_identity": 1.5}),
        ("style_oa_2_lr2e4_wc2_swd90_id30_e120", {"learning_rate": 2e-4, "w_color": 2.0, "w_swd": 90.0, "w_identity": 3.0}),
        ("style_oa_3_lr2e4_wc5_swd60_id30_e120", {"learning_rate": 2e-4, "w_color": 5.0, "w_swd": 60.0, "w_identity": 3.0}),
        ("style_oa_4_lr2e4_wc5_swd90_id15_e120", {"learning_rate": 2e-4, "w_color": 5.0, "w_swd": 90.0, "w_identity": 1.5}),
        ("style_oa_5_lr5e4_wc2_swd60_id30_e120", {"learning_rate": 5e-4, "w_color": 2.0, "w_swd": 60.0, "w_identity": 3.0}),
        ("style_oa_6_lr5e4_wc2_swd90_id15_e120", {"learning_rate": 5e-4, "w_color": 2.0, "w_swd": 90.0, "w_identity": 1.5}),
        ("style_oa_7_lr5e4_wc5_swd60_id15_e120", {"learning_rate": 5e-4, "w_color": 5.0, "w_swd": 60.0, "w_identity": 1.5}),
        ("style_oa_8_lr5e4_wc5_swd90_id30_e120", {"learning_rate": 5e-4, "w_color": 5.0, "w_swd": 90.0, "w_identity": 3.0}),
    ]

    run_bat = out_dir / "style_oa.bat"
    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f_bat.write("echo ==========================================\n")
        f_bat.write("echo Running Style OA Orthogonal Ablation (8 exps)\n")
        f_bat.write("echo ==========================================\n")

        for i, (name, cfg_delta) in enumerate(experiments, start=1):
            cfg = copy.deepcopy(base)
            cfg["training"]["learning_rate"] = float(cfg_delta["learning_rate"])
            cfg["training"]["min_learning_rate"] = float(cfg_delta["learning_rate"]) * 0.1
            cfg["loss"]["w_color"] = float(cfg_delta["w_color"])
            cfg["loss"]["w_swd"] = float(cfg_delta["w_swd"])
            cfg["loss"]["w_identity"] = float(cfg_delta["w_identity"])
            cfg.setdefault("checkpoint", {})
            cfg["checkpoint"]["save_dir"] = f"../{name}"

            cfg_filename = f"config_{name}.json"
            cfg_path = out_dir / cfg_filename
            with open(cfg_path, "w", encoding="utf-8") as f_cfg:
                json.dump(cfg, f_cfg, indent=4, ensure_ascii=False)
                f_cfg.write("\n")

            print(
                f"generated: {cfg_filename:74s} | "
                f"exp={i} mode={cfg['loss']['color_mode']:24s} "
                f"lr={cfg['training']['learning_rate']:.1e} "
                f"w_swd={cfg['loss']['w_swd']:.0f} "
                f"w_id={cfg['loss']['w_identity']:.2f} "
                f"w_color={cfg['loss']['w_color']:.2f}"
            )

            f_bat.write("echo.\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"echo Running Experiment {i}: {name}\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"uv run run.py --config {cfg_filename}\n")
            f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")

        f_bat.write("echo.\n")
        f_bat.write("echo Style OA ablation finished.\n")

    print("\nstyle_oa.bat has been generated.")


if __name__ == "__main__":
    generate_style_oa8()
