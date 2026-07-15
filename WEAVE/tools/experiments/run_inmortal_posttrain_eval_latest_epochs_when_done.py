from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[run_inmortal_posttrain_eval_latest_epochs_when_done] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _iter_proc_cmdlines() -> list[str]:
    proc_root = Path("/proc")
    rows: list[str] = []
    if proc_root.is_dir():
        for pid_dir in proc_root.iterdir():
            if not pid_dir.is_dir() or not pid_dir.name.isdigit():
                continue
            cmdline_path = pid_dir / "cmdline"
            try:
                raw = cmdline_path.read_bytes()
            except OSError:
                continue
            if not raw:
                continue
            text = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
            if text:
                rows.append(text)
        if rows:
            return rows
    result = subprocess.run(
        ["ps", "-eo", "args="],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _train_proc_alive(train_pattern: str) -> bool:
    pattern = str(train_pattern).strip()
    if not pattern:
        return False
    variants = {pattern}
    if not pattern.endswith(".json"):
        variants.add(pattern + ".json")
    if pattern.startswith("aaai2027_"):
        tail = pattern[len("aaai2027_") :]
        variants.add(tail)
        variants.add(f"configs/aaai2027/{tail}")
        if not tail.endswith(".json"):
            variants.add(tail + ".json")
            variants.add(f"configs/aaai2027/{tail}.json")
    for row in _iter_proc_cmdlines():
        if not row:
            continue
        if "run_inmortal_posttrain_eval_when_done.py" in row:
            continue
        if "run_inmortal_posttrain_eval_latest_epochs_when_done.py" in row:
            continue
        if any(variant and variant in row for variant in variants):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for a training pattern to finish, infer fresh epochs from the latest training log, then rerun eval on those epochs only.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-pattern", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--clip-hf-cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vae-decode-batch-size", type=int, default=16)
    parser.add_argument("--target-chunk-size", type=int, default=2)
    parser.add_argument("--code-root", default="mainline")
    parser.add_argument("--output-subdir", default="full_eval_fresh")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=43200)
    parser.add_argument("--require-seen-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-exit-settle-seconds", type=int, default=30)
    parser.add_argument("--infer-retries", type=int, default=12)
    parser.add_argument("--infer-retry-seconds", type=int, default=30)
    parser.add_argument("--save-generated-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-summary-grid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-enable-introstyle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--introstyle-style-bank-root", default="")
    parser.add_argument("--introstyle-model-id", default="")
    parser.add_argument("--introstyle-modelscope-id", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--introstyle-modelscope-cache-dir", default="")
    parser.add_argument("--introstyle-bank-limit-per-style", type=int, default=64)
    parser.add_argument("--introstyle-batch-size", type=int, default=2)
    parser.add_argument("--introstyle-topk", type=int, default=8)
    parser.add_argument("--introstyle-t", type=int, default=25)
    parser.add_argument("--introstyle-up-ft-index", type=int, default=1)
    parser.add_argument("--introstyle-ensemble-size", type=int, default=1)
    args = parser.parse_args()

    deadline = time.monotonic() + max(0, int(args.max_wait_seconds))
    seen_train_alive = False
    while True:
        alive = _train_proc_alive(str(args.train_pattern))
        seen_train_alive = seen_train_alive or alive
        print(
            f"[run_inmortal_posttrain_eval_latest_epochs_when_done] train_alive={alive} seen_train_alive={seen_train_alive} pattern={args.train_pattern}",
            flush=True,
        )
        if not alive and not (bool(args.require_seen_train) and not seen_train_alive):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for training to finish: {args.train_pattern}")
        time.sleep(max(1, int(args.poll_seconds)))

    settle_seconds = max(0, int(args.post_exit_settle_seconds))
    if settle_seconds > 0:
        print(
            f"[run_inmortal_posttrain_eval_latest_epochs_when_done] settling for {settle_seconds}s before fresh-epoch inference",
            flush=True,
        )
        time.sleep(settle_seconds)

    infer_cmd = [
        str(args.python_bin),
        str(Path(__file__).resolve().parent / "infer_fresh_epochs_from_latest_training_log.py"),
        "--run-dir",
        str(Path(args.run_dir).resolve()),
    ]
    epochs: list[str] = []
    infer_errors: list[str] = []
    infer_retries = max(1, int(args.infer_retries))
    infer_retry_seconds = max(1, int(args.infer_retry_seconds))
    for attempt in range(1, infer_retries + 1):
        result = subprocess.run(
            infer_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        stdout = result.stdout.strip()
        if result.returncode == 0:
            epochs = [token for token in stdout.split() if token.strip()]
            if epochs:
                print(
                    f"[run_inmortal_posttrain_eval_latest_epochs_when_done] inferred fresh epochs on attempt {attempt}: {' '.join(epochs)}",
                    flush=True,
                )
                break
            infer_errors.append(f"attempt {attempt}: empty epoch list")
        else:
            infer_errors.append(f"attempt {attempt}: rc={result.returncode} output={stdout}")
        if attempt < infer_retries:
            print(
                f"[run_inmortal_posttrain_eval_latest_epochs_when_done] fresh-epoch inference not ready on attempt {attempt}; sleeping {infer_retry_seconds}s",
                flush=True,
            )
            time.sleep(infer_retry_seconds)
    if not epochs:
        details = " | ".join(infer_errors[-3:]) if infer_errors else "no details"
        raise RuntimeError(f"fresh epoch inference failed after {infer_retries} attempts: {details}")
    cmd = [
        str(args.python_bin),
        str(Path(__file__).resolve().parent / "rerun_full_eval_for_run.py"),
        "--run-dir",
        str(Path(args.run_dir).resolve()),
        "--test-dir",
        str(args.test_dir),
        "--cache-dir",
        str(args.cache_dir),
        "--clip-hf-cache-dir",
        str(args.clip_hf_cache_dir),
        "--batch-size",
        str(int(args.batch_size)),
        "--vae-decode-batch-size",
        str(int(args.vae_decode_batch_size)),
        "--target-chunk-size",
        str(int(args.target_chunk_size)),
        "--profile-timing",
        "--code-root",
        str(args.code_root),
        "--output-subdir",
        str(args.output_subdir),
        "--skip-existing",
        "--epochs",
        *epochs,
    ]
    if bool(args.save_summary_grid):
        cmd.append("--save-summary-grid")
    else:
        cmd.append("--no-save-summary-grid")
    if bool(args.save_generated_images):
        cmd.append("--save-generated-images")
    else:
        cmd.append("--no-save-generated-images")
    if bool(args.eval_enable_introstyle):
        cmd.append("--eval-enable-introstyle")
        if str(args.introstyle_style_bank_root).strip():
            cmd += ["--introstyle-style-bank-root", str(args.introstyle_style_bank_root)]
        if str(args.introstyle_model_id).strip():
            cmd += ["--introstyle-model-id", str(args.introstyle_model_id)]
        if str(args.introstyle_modelscope_id).strip():
            cmd += ["--introstyle-modelscope-id", str(args.introstyle_modelscope_id)]
        if str(args.introstyle_modelscope_cache_dir).strip():
            cmd += ["--introstyle-modelscope-cache-dir", str(args.introstyle_modelscope_cache_dir)]
        cmd += ["--introstyle-bank-limit-per-style", str(int(args.introstyle_bank_limit_per_style))]
        cmd += ["--introstyle-batch-size", str(int(args.introstyle_batch_size))]
        cmd += ["--introstyle-topk", str(int(args.introstyle_topk))]
        cmd += ["--introstyle-t", str(int(args.introstyle_t))]
        cmd += ["--introstyle-up-ft-index", str(int(args.introstyle_up_ft_index))]
        cmd += ["--introstyle-ensemble-size", str(int(args.introstyle_ensemble_size))]
    else:
        cmd.append("--no-eval-enable-introstyle")
    _run(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
