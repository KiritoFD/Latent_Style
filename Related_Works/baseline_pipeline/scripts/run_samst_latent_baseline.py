from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent
SAMST_REPO = WORKSPACE_ROOT / "Related_Works" / "repos" / "SaMST-main"

DATASETS = {
    "legacy256_overfit50": {
        "latent_root": str((WORKSPACE_ROOT / "latent-256").resolve()),
        "style_names": ["photo", "monet", "vangogh", "cezanne", "Hayao"],
        "styles_dir": str((WORKSPACE_ROOT / "style_data" / "overfit50").resolve()),
        "image_size": 256,
        "style_size": 256,
    },
    "distinct5_512": {
        "latent_root": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train",
        "style_names": ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"],
        "styles_dir": "/mnt/i/wikiart_distinct5_samam_512_classview/test",
        "image_size": 512,
        "style_size": 512,
    },
}


def _read_optional_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _restore_optional_text(path: Path, content: str | None) -> None:
    if content is None:
        if path.exists():
            path.unlink()
        return
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-per-style", type=int, default=0)
    parser.add_argument("--vae-model", type=str, default="ema")
    parser.add_argument("--vae-cache-dir", type=str, default="")
    args = parser.parse_args()

    preset = DATASETS[args.dataset]
    out_root = (args.out_root or (PIPELINE_ROOT / "results" / f"samst_latent_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(exist_ok=True)
    meta = {
        "dataset": args.dataset,
        "preset": preset,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    style_single_root = out_root / "style_single"
    style_single_root.mkdir(parents=True, exist_ok=True)
    for style in preset["style_names"]:
        src_dir = Path(preset["styles_dir"]) / style
        first_file = sorted(p for p in src_dir.iterdir() if p.is_file())[0]
        dst_dir = style_single_root / style
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / first_file.name
        if not dst.exists():
            dst.write_bytes(first_file.read_bytes())

    train_dir = SAMST_REPO / "train_model" / "train2"
    train_yml = train_dir / "train.yml"
    original_train_yml = _read_optional_text(train_yml)
    cfg = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "dataset": str(preset["latent_root"]),
        "style_image": str(style_single_root / "") ,
        "style_latent_root": str(preset["latent_root"]),
        "save_model_dir": str((out_root / "checkpoints").resolve()),
        "image_size": int(preset["image_size"]),
        "style_size": int(preset["style_size"]),
        "cuda": 1,
        "seed": 7,
        "content_weight": 1e5,
        "style_weight": 1e10,
        "ae_weight": 1e3,
        "lr": 0.001,
        "weight_decay": 0.5,
        "step_size": 25,
        "save_interval": int(args.epochs),
        "log_interval": 10,
        "checkpoint_interval": 100,
        "checkpoint_model_dir": None,
        "begin_checkpoint": None,
        "begin_epoch": None,
        "max_steps": int(args.max_steps),
        "step_model_name_template": "step_{step:06d}.model",
        "latent_channels": 4,
        "latent_scaling_factor": 0.18215,
        "vae_model": str(args.vae_model),
        "vae_cache_dir": str(args.vae_cache_dir),
        "max_train_per_style": int(args.max_train_per_style),
    }
    train_yml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    cmd = [sys.executable, str(train_dir / "train_latent.py")]
    log_path = out_root / "logs" / "train.log"
    started = time.time()
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n=== START {datetime.now().isoformat()} ===\n")
            log.write("CMD: " + " ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(train_dir), stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
            elapsed = time.time() - started
            log.write(f"\n=== END rc={proc.returncode} elapsed_sec={elapsed:.3f} ===\n")
        return proc.returncode
    finally:
        _restore_optional_text(train_yml, original_train_yml)


if __name__ == "__main__":
    raise SystemExit(main())
