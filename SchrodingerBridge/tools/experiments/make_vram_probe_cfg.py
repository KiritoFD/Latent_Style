from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a short-step VRAM probe config from a training config.")
    parser.add_argument("--input", required=True, help="Source config.json path")
    parser.add_argument("--output", required=True, help="Target probe config path")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--stop-after-steps", type=int, default=20)
    parser.add_argument("--num-epochs", type=int, default=1)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    with src.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    training = cfg.setdefault("training", {})
    training["batch_size"] = int(args.batch_size)
    training["num_epochs"] = int(args.num_epochs)
    training["save_interval"] = 1
    training["stop_after_global_steps"] = int(args.stop_after_steps)
    training["resume_optimizer"] = False
    training["resume_training_state"] = False
    training["resume_prefer_local_checkpoint"] = False
    training["full_eval_each_epoch"] = False
    training["full_eval_defer_until_training_end"] = False
    training["full_eval_stop_on_convergence"] = False

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
