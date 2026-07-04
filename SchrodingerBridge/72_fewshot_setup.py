"""Few-shot experiment: data prep + checkpoint expansion + config generation.

Uses T11 (local SOTA, clip=0.7213, lpips=0.2868) as base checkpoint.
Creates few-shot datasets from existing fewshot8 latent data by subsetting.
"""
import os, sys, json, random, shutil, torch
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge")
PYTHON = sys.executable

T11_CKPT = ROOT / "exp" / "FCSB" / "local_t" / "630_local_t11_stochastic_dwt_p08" / "epoch_0005.pt"
T11_CONFIG = ROOT / "exp" / "FCSB" / "local_t" / "630_local_t11_stochastic_dwt_p08" / "config.json"

DISTINCT5_TRAIN = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512_latents_ema\train")
FEWSHOT8_TRAIN = Path(r"G:\GitHub\Latent_Style\Dataset\fewshot8_512_latents_ema\train")
TEST_DIR_BASE = Path(r"G:\GitHub\Latent_Style\Dataset\fewshot8_512_latents_ema\test")
DISTINCT5_TEST = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test")

OUTPUT_BASE = ROOT / "exp" / "72_fewshot"
CONFIG_DIR = ROOT / "configs" / "72_fewshot"
EXPANDED_CKPT_DIR = OUTPUT_BASE / "expanded_ckpt"

BASE_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
NEW_STYLE_CANDIDATES = ["Expressionism", "Post_Impressionism", "Realism"]
SHOT_COUNTS = [1, 6, 10, 30, 50]
SEED = 42


def prepare_fewshot_dataset(new_styles, shots, data_dir):
    """Create a train dir with base5 as symlinks and new styles as subset of .pt files."""
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    all_styles = BASE_STYLES + new_styles
    
    # Base 5 styles: symlink to distinct5 train
    for style in BASE_STYLES:
        link = train_dir / style
        target = DISTINCT5_TRAIN / style
        if link.exists() or link.is_symlink():
            if link.is_symlink():
                link.unlink()
            else:
                shutil.rmtree(str(link))
        os.symlink(str(target), str(link))
    
    # New styles: copy subset of .pt files from fewshot8 train
    for style in new_styles:
        src_dir = FEWSHOT8_TRAIN / style
        dst_dir = train_dir / style
        if dst_dir.is_symlink():
            dst_dir.unlink()
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        all_files = sorted(src_dir.glob("*.pt"))
        total = len(all_files)
        rng = random.Random(SEED)
        selected = sorted(rng.sample(all_files, min(shots, total)))
        for f in selected:
            dst = dst_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
        print(f"    {style}: {len(selected)}/{total} shots")
    
    # Remove any stale packed cache so dataset rebuilds it
    cache_dir = train_dir / ".latent_cache"
    if cache_dir.exists():
        shutil.rmtree(str(cache_dir))
    
    # Test dir: symlink base5 from distinct5 test, copy new styles from fewshot8 test
    test_dir = data_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for style in BASE_STYLES:
        link = test_dir / style
        target = DISTINCT5_TEST / style
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            shutil.rmtree(str(link))
        os.symlink(str(target), str(link))
    for style in new_styles:
        src = TEST_DIR_BASE / style
        dst = test_dir / style
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            shutil.rmtree(str(dst))
        shutil.copytree(str(src), str(dst))
    
    return data_dir


def expand_checkpoint(num_styles):
    """Expand T11 checkpoint to num_styles using tools/expand_checkpoint_num_styles.py."""
    dst = EXPANDED_CKPT_DIR / f"t11_expanded_{num_styles}styles.pt"
    if dst.exists():
        print(f"  Already exists: {dst}")
        return dst
    EXPANDED_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    import subprocess
    cmd = [PYTHON, str(ROOT / "tools" / "expand_checkpoint_num_styles.py"),
           "--src", str(T11_CKPT), "--dst", str(dst),
           "--old-num-styles", "5", "--new-num-styles", str(num_styles)]
    print(f"  Expanding 5 -> {num_styles} styles...")
    subprocess.run(cmd, check=True)
    return dst


def generate_config(exp_name, num_styles, new_styles, shots, data_root, expanded_ckpt):
    """Generate training config for a few-shot experiment."""
    with open(T11_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    all_styles = BASE_STYLES + new_styles
    
    cfg["model"]["num_styles"] = num_styles
    cfg["data"]["style_subdirs"] = all_styles
    cfg["data"]["data_root"] = str(data_root / "train")
    cfg["data"]["latent_cache_dir"] = ""
    cfg["data"]["latent_cache_mode"] = "packed"
    cfg["data"]["pairing_cache_path"] = ""
    cfg["data"]["pairing_cache_cross_only"] = True
    
    cfg["training"]["num_epochs"] = 10
    cfg["training"]["save_interval"] = 1
    cfg["training"]["full_eval_each_epoch"] = True
    cfg["training"]["full_eval_defer_until_training_end"] = False
    cfg["training"]["freeze_mode"] = "tokenizer_only"
    cfg["training"]["freeze_reinit_trainable"] = False
    cfg["training"]["learning_rate"] = 0.0002
    cfg["training"]["resume_checkpoint"] = str(expanded_ckpt)
    cfg["training"]["resume_model_strict"] = False
    cfg["training"]["resume_optimizer"] = False
    cfg["training"]["resume_training_state"] = False
    cfg["training"]["test_image_dir"] = str(data_root / "test")
    cfg["training"]["full_eval_cache_dir"] = "G:/GitHub/Latent_Style/eval_cache"
    cfg["training"]["full_eval_clip_hf_cache_dir"] = "G:/GitHub/Latent_Style/eval_cache/hf"
    
    cfg["checkpoint"]["save_dir"] = f"./exp/72_fewshot/{exp_name}"
    
    cfg["ablation"] = {
        "name": exp_name,
        "axis": "72_fewshot",
        "notes": f"5+{len(new_styles)} styles, {shots} shots, freeze_mode=tokenizer_only, base=T11",
        "new_styles": new_styles,
        "shots": shots,
    }
    
    out_path = CONFIG_DIR / f"{exp_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  Config: {out_path.name}")
    return out_path


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    if not T11_CKPT.exists():
        print(f"ERROR: T11 checkpoint not found at {T11_CKPT}")
        sys.exit(1)
    print(f"Base checkpoint: {T11_CKPT}")
    print(f"Base config: {T11_CONFIG}")
    
    # Step 1: Expand checkpoints
    print("\n=== Step 1: Expand checkpoints ===")
    expanded_ckpts = {}
    for n_new in [1, 2, 3]:
        num_styles = 5 + n_new
        expanded_ckpts[num_styles] = expand_checkpoint(num_styles)
    
    # Step 2: Prepare few-shot datasets + configs
    print("\n=== Step 2: Prepare datasets + configs ===")
    experiments = []
    for n_new in [1, 2, 3]:
        new_styles = NEW_STYLE_CANDIDATES[:n_new]
        num_styles = 5 + n_new
        for shots in SHOT_COUNTS:
            exp_name = f"5p{n_new}_shot{shots:02d}"
            data_dir = OUTPUT_BASE / "data" / exp_name
            print(f"\n  {exp_name}: 5+{n_new} styles, {shots} shots")
            prepare_fewshot_dataset(new_styles, shots, data_dir)
            cfg_path = generate_config(exp_name, num_styles, new_styles, shots, data_dir, expanded_ckpts[num_styles])
            experiments.append({"name": exp_name, "config": str(cfg_path), "shots": shots, "n_new": n_new})
    
    # Step 3: Create batch training runner
    print(f"\n=== Step 3: Batch runner ===")
    runner_path = OUTPUT_BASE / "run_all.py"
    runner_code = [
        '"""Batch runner for few-shot experiments."""',
        'import os, sys, subprocess, time',
        'from pathlib import Path',
        '',
        'ROOT = Path(r"G:\\GitHub\\Latent_Style\\SchrodingerBridge")',
        'PYTHON = sys.executable',
        'RUN_SCRIPT = str(ROOT / "src" / "run.py")',
        'CONFIG_DIR = ROOT / "configs" / "72_fewshot"',
        'LOG_DIR = ROOT / "exp" / "72_fewshot" / "logs"',
        'LOG_DIR.mkdir(parents=True, exist_ok=True)',
        '',
        'experiments = sorted(CONFIG_DIR.glob("*.json"))',
        'total = len(experiments)',
        'print(f"Running {total} few-shot experiments...")',
        '',
        'for i, cfg_path in enumerate(experiments, 1):',
        '    name = cfg_path.stem',
        '    log_path = LOG_DIR / f"{name}.log"',
        '    exp_dir = ROOT / "exp" / "72_fewshot" / name',
        '    done_marker = exp_dir / "epoch_0010.pt"',
        '    if done_marker.exists():',
        '        print(f"[{i}/{total}] SKIP {name} (already done)")',
        '        continue',
        '    print(f"[{i}/{total}] START {name}")',
        '    t0 = time.time()',
        '    with open(log_path, "w", encoding="utf-8") as log_f:',
        '        result = subprocess.run(',
        '            [PYTHON, RUN_SCRIPT, "--config", str(cfg_path)],',
        '            stdout=log_f, stderr=subprocess.STDOUT,',
        '            timeout=1800,',
        '        )',
        '    elapsed = time.time() - t0',
        '    status = "SUCCESS" if result.returncode == 0 else f"FAILED rc={result.returncode}"',
        '    print(f"[{i}/{total}] {name} {status} ({elapsed:.0f}s)")',
        '',
        'print("All done!")',
    ]
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write("\n".join(runner_code) + "\n")
    print(f"  Runner: {runner_path}")
    
    print(f"\n=== Summary ===")
    print(f"  {len(experiments)} experiments to run")
    print(f"  Run: python {runner_path}")
    print(f"  Or individual: python src/run.py --config configs/72_fewshot/<name>.json")


if __name__ == "__main__":
    main()
