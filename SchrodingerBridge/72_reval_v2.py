import json, sys, subprocess, shutil
from pathlib import Path

ROOT = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge")
PYTHON = sys.executable
EVAL_SCRIPT = ROOT / "src" / "utils" / "run_evaluation.py"
EXP_BASE = Path(r"G:\GitHub\Latent_Style\exp\72_fewshot")
CACHE_BASE = Path(r"G:\GitHub\Latent_Style\eval_cache")

experiments = sorted([d.name for d in EXP_BASE.iterdir() if d.name.startswith("5p") and d.is_dir()])

for exp_name in experiments:
    exp_dir = EXP_BASE / exp_name
    cfg_path = ROOT / "configs" / "72_fewshot" / (exp_name + ".json")
    if not cfg_path.exists():
        print(f"SKIP {exp_name}: no config")
        continue
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    ckpts = sorted(exp_dir.glob("epoch_*.pt"))
    if not ckpts:
        print(f"SKIP {exp_name}: no checkpoints")
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

    print(f"\n{'='*60}")
    print(f"Evaluating {exp_name}/{epoch_name}...")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, timeout=1800)
        status = "OK" if result.returncode == 0 else f"FAILED rc={result.returncode}"
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
    print(f"  {status}")

print("\nAll re-eval done!")
