from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]


STYLE_NAMES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]


def _run(cmd: list[str], *, cwd: Path | None = None) -> int:
    print("[watch_wikiarts5_samst_eval_bundle] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=None if cwd is None else str(cwd), check=False)
    return int(proc.returncode)


def _bundle_process_alive(*, run_root: Path, epoch: int) -> bool:
    marker = f"run_samst_distinct5_eval_bundle.py --run-root {str(run_root)} --epochs {int(epoch)}"
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -eq 'python.exe' -and "
                f"$_.CommandLine -like '*{marker}*' }} | "
                "Select-Object -ExpandProperty ProcessId"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return any(line.strip().isdigit() for line in proc.stdout.splitlines())


def _available_epoch_numbers(run_root: Path, style_names: list[str]) -> list[int]:
    per_style: list[set[int]] = []
    for style in style_names:
        style_dir = run_root / "checkpoints" / style
        epochs: set[int] = set()
        if style_dir.is_dir():
            for path in style_dir.glob("epoch_*.model"):
                digits = "".join(ch for ch in path.stem if ch.isdigit())
                if digits:
                    epochs.add(int(digits))
        per_style.append(epochs)
    if not per_style:
        return []
    common = set.intersection(*per_style) if per_style else set()
    return sorted(common)


def _per_style_epoch_counts(run_root: Path, style_names: list[str]) -> dict[str, int]:
    payload: dict[str, int] = {}
    for style in style_names:
        style_dir = run_root / "checkpoints" / style
        payload[style] = len(list(style_dir.glob("epoch_*.model"))) if style_dir.is_dir() else 0
    return payload


def _read_curve_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rebuild_curve(eval_root: Path) -> Path | None:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(eval_root.glob("eval_epoch*/epoch_*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        analysis = payload.get("analysis") or {}
        transfer = analysis.get("style_transfer_ability") or {}
        full = analysis.get("all_pairs_overview") or {}
        timings = payload.get("timings_sec") or {}
        epoch_digits = "".join(ch for ch in summary_path.parent.name if ch.isdigit())
        rows.append(
            {
                "epoch": summary_path.parent.name,
                "epoch_num": int(epoch_digits) if epoch_digits else -1,
                "transfer_clip_style": transfer.get("clip_style"),
                "transfer_content_lpips": transfer.get("content_lpips"),
                "full_clip_style": full.get("clip_style"),
                "full_content_lpips": full.get("content_lpips"),
                "wall_total_seconds": timings.get("wall_total"),
                "summary_path": str(summary_path),
            }
        )
    if not rows:
        return None
    curve_csv = eval_root / "clip_lpips_curve.csv"
    with curve_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return curve_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch a wikiarts5 SaMST run and trigger CLIP-S/LPIPS eval every N common epochs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--epoch-interval", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--style-names", default=",".join(STYLE_NAMES))
    parser.add_argument("--test-root", type=Path, default=Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test"))
    parser.add_argument("--max-src-per-style", type=int, default=30)
    parser.add_argument("--resize-content", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-target-chunk-size", type=int, default=1)
    parser.add_argument("--eval-vae-decode-batch-size", type=int, default=0)
    parser.add_argument("--eval-image-save-workers", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=0)
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    style_names = [x.strip() for x in str(args.style_names).split(",") if x.strip()]
    eval_root = run_root / "eval_bundle"
    eval_root.mkdir(parents=True, exist_ok=True)
    watch_log = run_root / "watch_eval_every5.log"
    seen_epochs: set[int] = set()
    cycles = 0

    while True:
        common_epochs = [ep for ep in _available_epoch_numbers(run_root, style_names) if ep > 0 and ep % int(args.epoch_interval) == 0]
        per_style_counts = _per_style_epoch_counts(run_root, style_names)
        with watch_log.open("a", encoding="utf-8") as log:
            log.write(
                json.dumps(
                    {
                        "event": "poll",
                        "common_epochs": common_epochs,
                        "per_style_epoch_counts": per_style_counts,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        for epoch in common_epochs:
            if epoch in seen_epochs:
                continue
            summary_json = eval_root / f"eval_epoch{epoch}" / f"epoch_{epoch:04d}" / "summary.json"
            if summary_json.is_file():
                seen_epochs.add(epoch)
                continue
            if _bundle_process_alive(run_root=run_root, epoch=epoch):
                with watch_log.open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"event": "eval_inflight", "epoch": epoch}, ensure_ascii=False) + "\n")
                continue
            with watch_log.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"event": "launch_eval", "epoch": epoch, "time": time.time()}, ensure_ascii=False) + "\n")
            rc = _run(
                [
                    sys.executable,
                    str(WORKSPACE / "Related_Works" / "baseline_pipeline" / "scripts" / "run_samst_distinct5_eval_bundle.py"),
                    "--run-root",
                    str(run_root),
                    "--epochs",
                    str(epoch),
                    "--test-root",
                    str(Path(args.test_root).resolve()),
                    "--style-names",
                    ",".join(style_names),
                    "--max-src-per-style",
                    str(int(args.max_src_per_style)),
                    "--resize-content",
                    str(int(args.resize_content)),
                    "--eval-batch-size",
                    str(int(args.eval_batch_size)),
                    "--eval-target-chunk-size",
                    str(int(args.eval_target_chunk_size)),
                    "--eval-vae-decode-batch-size",
                    str(int(args.eval_vae_decode_batch_size)),
                    "--eval-image-save-workers",
                    str(int(args.eval_image_save_workers)),
                ]
            )
            if rc == 0:
                seen_epochs.add(epoch)
                curve_csv = _rebuild_curve(eval_root)
                with watch_log.open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"event": "eval_done", "epoch": epoch, "curve_csv": "" if curve_csv is None else str(curve_csv)}, ensure_ascii=False) + "\n")
            else:
                with watch_log.open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"event": "eval_failed", "epoch": epoch, "rc": rc}, ensure_ascii=False) + "\n")
        cycles += 1
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
