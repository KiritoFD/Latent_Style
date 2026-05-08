from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
BASE_CONFIG_PATH = REPO_ROOT / "experiments" / "semantic_overfit_small" / "A_baseline" / "config.json"
OUT_ROOT = REPO_ROOT / "experiments" / "semantic_full40_kinetic_sweep"


KINETIC_RUNS = [
    ("kin40_10", 1.0),
    ("kin40_02", 0.2),
    ("kin40_00", 0.0),
]


FULL_STYLE_SUBDIRS = ["photo", "Hayao", "monet", "vangogh", "cezanne"]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as f:
        base = json.load(f)

    base["model"]["num_styles"] = len(FULL_STYLE_SUBDIRS)
    base["data"]["style_subdirs"] = FULL_STYLE_SUBDIRS
    base["data"]["data_root"] = "../latent-256"
    base["data"]["balance_target_styles_per_batch"] = True
    base["data"]["virtual_length_multiplier"] = 1
    base["data"]["identity_ratio"] = 0.0

    base["training"]["batch_size"] = 96
    base["training"]["num_workers"] = 0
    base["training"]["shuffle"] = False
    base["training"]["persistent_workers"] = False
    base["training"]["num_epochs"] = 40
    base["training"]["save_interval"] = 20
    base["training"]["log_interval"] = 20
    base["training"]["full_eval_batch_size"] = 2
    base["training"]["test_image_dir"] = "../style_data/overfit50"

    # Keep the validated semantic-SBMF equation from the single-pair test.
    base["bridge"]["w_color"] = 0.0
    base["bridge"]["w_repulsive"] = 0.0
    base["bridge"]["w_nce"] = 0.0
    base["bridge"]["w_low_freq"] = 1.0
    base["bridge"]["w_cycle"] = 1.0
    base["bridge"]["terminal_swd_weight"] = 0.1
    base["bridge"]["semantic_swd_num_projections"] = 16
    base["bridge"]["w_flow"] = 0.0

    manifest: list[dict[str, object]] = []
    for name, kinetic in KINETIC_RUNS:
        cfg = json.loads(json.dumps(base))
        cfg["bridge"]["w_kinetic"] = kinetic
        cfg["checkpoint"]["save_dir"] = f"./experiments/semantic_full40_kinetic_sweep/{name}"

        cfg_path = OUT_ROOT / f"{name}.json"
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")

        manifest.append(
            {
                "name": name,
                "config": str(cfg_path),
                "w_kinetic": kinetic,
                "save_dir": cfg["checkpoint"]["save_dir"],
            }
        )

    with (OUT_ROOT / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    train_bat = REPO_ROOT / "run_semantic_full40_kinetic_sweep.bat"
    eval_bat = REPO_ROOT / "eval_semantic_full40_kinetic_sweep.bat"

    train_lines = [
        "@echo off",
        "setlocal",
        "cd /d %~dp0",
        'for %%N in (kin40_10 kin40_02 kin40_00) do (',
        '  echo ==================================================',
        '  echo Training %%N',
        '  python run.py --config "experiments\\semantic_full40_kinetic_sweep\\%%N.json"',
        "  if errorlevel 1 exit /b %errorlevel%",
        ")",
        "endlocal",
    ]
    train_bat.write_text("\n".join(train_lines) + "\n", encoding="utf-8")

    eval_lines = [
        "@echo off",
        "setlocal",
        "cd /d %~dp0",
        'for %%N in (kin40_10 kin40_02 kin40_00) do (',
        '  if exist "experiments\\semantic_full40_kinetic_sweep\\%%N\\epoch_0040.pt" (',
        '    echo ==================================================',
        '    echo Evaluating %%N',
        '    python run_evaluation.py "experiments\\semantic_full40_kinetic_sweep\\%%N" --output "experiments\\semantic_full40_kinetic_sweep\\full_eval\\%%N" --batch_size 2 --force',
        "    if errorlevel 1 exit /b %errorlevel%",
        "  )",
        ")",
        "endlocal",
    ]
    eval_bat.write_text("\n".join(eval_lines) + "\n", encoding="utf-8")

    print(f"Wrote configs to: {OUT_ROOT}")
    print(f"Wrote train script: {train_bat}")
    print(f"Wrote eval script: {eval_bat}")


if __name__ == "__main__":
    main()
