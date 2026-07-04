import json, sys, subprocess, shutil
from pathlib import Path

ROOT = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge")
PYTHON = sys.executable
EVAL_SCRIPT = ROOT / "src" / "utils" / "run_evaluation.py"
EXP_BASE = ROOT / "exp" / "72_fewshot"

experiments = sorted([d.name for d in EXP_BASE.iterdir() if d.name.startswith("5p") and d.is_dir()])

for exp_name in experiments:
    exp_dir = EXP_BASE / exp_name
    cfg_path = ROOT / "configs" / "72_fewshot" / (exp_name + ".json")
    if not cfg_path.exists():
        continue
    cfg = json.load(open(cfg_path, "r", encoding="utf-8"))

    ckpts = sorted(exp_dir.glob("epoch_*.pt"))
    if not ckpts:
        continue
    last_ckpt = ckpts[-1]
    epoch_name = last_ckpt.stem

    eval_dir = exp_dir / "full_eval" / epoch_name
    if eval_dir.exists():
        shutil.rmtree(str(eval_dir))
    out_dir = exp_dir / "full_eval" / epoch_name

    cmd = [
        PYTHON, str(EVAL_SCRIPT),
        "--checkpoint", str(last_ckpt),
        "--output", str(out_dir),
        "--test_dir", cfg["training"]["test_image_dir"],
        "--cache_dir", cfg["training"]["full_eval_cache_dir"],
        "--clip_hf_cache_dir", cfg["training"]["full_eval_clip_hf_cache_dir"],
        "--batch_size", "2",
        "--force_regen_ref_cache",
    ]

    print(f"Evaluating {exp_name}/{epoch_name}...")
    result = subprocess.run(cmd, timeout=1800)
    status = "OK" if result.returncode == 0 else f"FAILED rc={result.returncode}"
    print(f"  {status}")

print("All re-eval done!")
