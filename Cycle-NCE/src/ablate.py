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
        "config_tmp_batch*.json",
        "run_color_r16_e60.bat",
        "run_color_ablation_anchor_4.bat",
        "anchor4.bat",
        "weight.bat",
    ]
    removed = 0
    for pattern in old_patterns:
        for p in out_dir.glob(pattern):
            if p.is_file():
                p.unlink(missing_ok=True)
                removed += 1
    if removed > 0:
        print(f"cleaned old files: {removed}")


def generate_weight8() -> None:
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
    base["training"]["num_epochs"] = 60
    base["training"]["full_eval_interval"] = 30
    base["training"]["save_interval"] = 30
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
    base["loss"]["w_swd"] = 30.0
    base["loss"]["w_identity"] = 2.0
    base["loss"]["w_color"] = 2.0
    base["loss"]["w_delta_tv"] = 0.1
    base["loss"]["color_eps"] = 1e-6
    base["loss"]["color_legacy_pool"] = 4
    base["loss"]["color_mode"] = "latent_decoupled_adain"

    base.setdefault("data", {})
    # image_root is not used in the train/run path for latent dataset training.
    base["data"].pop("image_root", None)

    experiments = [
        ("weight_exp1_latent_adain_swd30_tv01_id20_r16_e60", {"color_mode": "latent_decoupled_adain", "w_swd": 30.0, "w_delta_tv": 0.1, "w_identity": 2.0}),
        ("weight_exp2_latent_adain_swd30_tv00_id40_r16_e60", {"color_mode": "latent_decoupled_adain", "w_swd": 30.0, "w_delta_tv": 0.0, "w_identity": 4.0}),
        ("weight_exp3_latent_adain_swd60_tv01_id20_r16_e60", {"color_mode": "latent_decoupled_adain", "w_swd": 60.0, "w_delta_tv": 0.1, "w_identity": 2.0}),
        ("weight_exp4_latent_adain_swd60_tv00_id40_r16_e60", {"color_mode": "latent_decoupled_adain", "w_swd": 60.0, "w_delta_tv": 0.0, "w_identity": 4.0}),
        ("weight_exp5_pseudo_hist_swd30_tv01_id20_r16_e60", {"color_mode": "pseudo_rgb_hist", "w_swd": 30.0, "w_delta_tv": 0.1, "w_identity": 2.0}),
        ("weight_exp6_pseudo_hist_swd30_tv00_id40_r16_e60", {"color_mode": "pseudo_rgb_hist", "w_swd": 30.0, "w_delta_tv": 0.0, "w_identity": 4.0}),
        ("weight_exp7_pseudo_hist_swd60_tv01_id20_r16_e60", {"color_mode": "pseudo_rgb_hist", "w_swd": 60.0, "w_delta_tv": 0.1, "w_identity": 2.0}),
        ("weight_exp8_pseudo_hist_swd60_tv00_id40_r16_e60", {"color_mode": "pseudo_rgb_hist", "w_swd": 60.0, "w_delta_tv": 0.0, "w_identity": 4.0}),
    ]

    run_bat = out_dir / "weight.bat"
    with open(run_bat, "w", encoding="utf-8") as f_bat:
        f_bat.write("@echo off\n")
        f_bat.write("setlocal\n")
        f_bat.write("cd /d %~dp0\n")
        f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")
        f_bat.write("echo ==========================================\n")
        f_bat.write("echo Running Weight 2x2x2 Ablation (8 exps)\n")
        f_bat.write("echo ==========================================\n")

        for i, (name, loss_delta) in enumerate(experiments, start=1):
            cfg = copy.deepcopy(base)
            cfg["loss"].update(loss_delta)
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
                f"w_swd={cfg['loss']['w_swd']:.0f} "
                f"w_id={cfg['loss']['w_identity']:.2f} "
                f"w_color={cfg['loss']['w_color']:.2f} "
                f"w_tv={cfg['loss']['w_delta_tv']:.2f}"
            )

            f_bat.write("echo.\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"echo Running Experiment {i}: {name}\n")
            f_bat.write("echo ------------------------------------------\n")
            f_bat.write(f"uv run run.py --config {cfg_filename}\n")
            f_bat.write("if %errorlevel% neq 0 exit /b %errorlevel%\n")

        f_bat.write("echo.\n")
        f_bat.write("echo Weight ablation finished.\n")

    print("\nweight.bat has been generated.")


if __name__ == "__main__":
    generate_weight8()
