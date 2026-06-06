from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent
SAMAM_ROOT = WORKSPACE_ROOT / "Related_Works" / "repos" / "SaMam"

DATASETS = {
    "legacy256_overfit50": {
        "latent_content_root": str((WORKSPACE_ROOT / "latent-256").resolve()),
        "latent_style_root": str((WORKSPACE_ROOT / "latent-256").resolve()),
        "val_content_root": str((WORKSPACE_ROOT / "latent-256").resolve()),
        "val_style_root": str((WORKSPACE_ROOT / "latent-256").resolve()),
        "eval_root": str((WORKSPACE_ROOT / "style_data" / "overfit50").resolve()),
        "styles": ["photo", "monet", "vangogh", "cezanne", "Hayao"],
        # RGB SaMam used patch_size=8 on 256/512 images, but latent inputs are already
        # 8x smaller spatially. Keeping 8 here collapses 32x32 latents down to 4x4 tokens.
        "patch_size": 4,
    },
    "distinct5_512": {
        "latent_content_root": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train",
        "latent_style_root": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train",
        "val_content_root": "/mnt/i/wikiart_distinct5_latents_512_ema_test",
        "val_style_root": "/mnt/i/wikiart_distinct5_latents_512_ema_test",
        "eval_root": "/mnt/i/wikiart_distinct5_samam_512_classview/test",
        "styles": ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"],
        # 512px EMA latents are 4x64x64. Using patch_size=8 reduces them to an 8x8 token map
        # and the decoder can only recover coarse color blobs. patch_size=4 preserves a 16x16 grid.
        "patch_size": 4,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=500)
    parser.add_argument("--max-train-content-per-style", type=int, default=0)
    parser.add_argument("--max-train-style-per-style", type=int, default=0)
    parser.add_argument("--max-val-content-per-style", type=int, default=1)
    parser.add_argument("--max-val-style-per-style", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", type=int, default=0)
    parser.add_argument("--identity-gradient-checkpointing", type=int, default=0)
    parser.add_argument("--vae-gradient-checkpointing", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--num-sanity-val-steps", type=int, default=2)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--vae-model", type=str, default="ema")
    parser.add_argument("--vae-cache-dir", type=str, default="")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    args = parser.parse_args()

    preset = DATASETS[args.dataset]
    out_root = (args.out_root or (PIPELINE_ROOT / "results" / f"samam_latent_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "train.log"
    meta = {
        "dataset": args.dataset,
        "preset": preset,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "limit_val_batches": args.limit_val_batches,
        "num_sanity_val_steps": args.num_sanity_val_steps,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "created_at": datetime.now().isoformat(),
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cmd = [
        sys.executable,
        str(SAMAM_ROOT / "TRAIN" / "train_SaMam_latent.py"),
        "--log-dir",
        str(out_root),
        "--iterations",
        str(args.iterations),
        "--val-interval",
        str(args.val_interval),
        "--latent-content-root",
        str(preset["latent_content_root"]),
        "--latent-style-root",
        str(preset["latent_style_root"]),
        "--val-content-root",
        str(preset["val_content_root"]),
        "--val-style-root",
        str(preset["val_style_root"]),
        "--batch-size",
        str(args.batch_size),
        "--precision",
        str(args.precision),
        "--checkpoint-every-n-steps",
        str(args.checkpoint_every_n_steps),
        "--max-train-content-per-style",
        str(args.max_train_content_per_style),
        "--max-train-style-per-style",
        str(args.max_train_style_per_style),
        "--max-val-content-per-style",
        str(args.max_val_content_per_style),
        "--max-val-style-per-style",
        str(args.max_val_style_per_style),
        "--num-workers",
        str(args.num_workers),
        "--pin-memory",
        str(args.pin_memory),
        "--gradient-checkpointing",
        str(args.gradient_checkpointing),
        "--identity-gradient-checkpointing",
        str(args.identity_gradient_checkpointing),
        "--vae-gradient-checkpointing",
        str(args.vae_gradient_checkpointing),
        "--limit-val-batches",
        str(args.limit_val_batches),
        "--num-sanity-val-steps",
        str(args.num_sanity_val_steps),
        "--accumulate-grad-batches",
        str(args.accumulate_grad_batches),
        "--vae-model",
        str(args.vae_model),
        "--vae-cache-dir",
        str(args.vae_cache_dir),
        "--patch-size",
        str(preset["patch_size"]),
        "--gpus",
        *[str(g) for g in args.gpus],
    ]
    started = time.time()
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n=== START {datetime.now().isoformat()} ===\n")
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(SAMAM_ROOT), stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
        elapsed = time.time() - started
        log.write(f"\n=== END rc={proc.returncode} elapsed_sec={elapsed:.3f} ===\n")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
