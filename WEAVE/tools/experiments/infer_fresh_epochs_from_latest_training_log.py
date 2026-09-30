from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer fresh epoch numbers from the latest training_*.csv timestamp and checkpoint mtimes.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    logs_dir = run_dir / "logs"
    logs = sorted(logs_dir.glob("training_*.csv"))
    if not logs:
        raise FileNotFoundError(f"No training_*.csv found under {logs_dir}")
    latest = logs[-1]
    stamp = latest.stem.split("_", 1)[1]
    start_dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    cutoff = start_dt.timestamp()

    epochs: list[int] = []
    for ckpt in sorted(run_dir.glob("epoch_*.pt")):
        try:
            epoch = int(ckpt.stem.split("_", 1)[1])
        except Exception:
            continue
        if ckpt.stat().st_mtime >= cutoff:
            epochs.append(epoch)
    if not epochs:
        raise RuntimeError(f"No fresh checkpoints found under {run_dir} using cutoff from {latest.name}")
    print(" ".join(str(ep) for ep in epochs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
