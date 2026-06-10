from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = Path(
    r"G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_wikiarts5_wsl_20260610_172206"
)
DEFAULT_STDOUT_LOG = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\samst_wikiarts5_segmented.stdout.log")
DEFAULT_STDERR_LOG = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\samst_wikiarts5_segmented.stderr.log")
STYLE_NAMES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]


def _to_wsl_mount(path: Path) -> str:
    text = str(path)
    if len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        remainder = text[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{remainder}" if remainder else f"/mnt/{drive}"
    return text.replace("\\", "/")


def _json_flag(path: Path, *, key: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    value = payload
    for part in str(key).split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return bool(value)


def _wsl_has_process(*, distro: str, match_text: str) -> bool:
    proc = subprocess.run(
        [
            "wsl",
            "-d",
            str(distro),
            "bash",
            "-lc",
            f"ps -eo cmd | grep -F -- {subprocess.list2cmdline([str(match_text)])} | grep -v grep || true",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return bool(proc.stdout.strip())


def _available_common_epochs(run_root: Path, style_names: list[str]) -> list[int]:
    per_style: list[set[int]] = []
    for style in style_names:
        epochs: set[int] = set()
        style_dir = run_root / "checkpoints" / style
        if style_dir.is_dir():
            for path in style_dir.glob("epoch_*.model"):
                digits = "".join(ch for ch in path.stem if ch.isdigit())
                if digits:
                    epochs.add(int(digits))
        per_style.append(epochs)
    if not per_style:
        return []
    return sorted(set.intersection(*per_style))


def _launch_segment(
    *,
    result_root: Path,
    stdout_log: Path,
    stderr_log: Path,
    wsl_distro: str,
    wsl_python: str,
    data_root: str,
    target_epoch: int,
    save_interval: int,
    batch_size: int,
    image_size: int,
    style_size: int,
    styles: str,
) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_wsl_samst_wikiarts5.py"),
        "--wsl-distro",
        str(wsl_distro),
        "--wsl-python",
        str(wsl_python),
        "--data-root",
        str(data_root),
        "--out-root",
        _to_wsl_mount(result_root),
        "--styles",
        str(styles),
        "--epochs",
        str(int(target_epoch)),
        "--batch-size",
        str(int(batch_size)),
        "--image-size",
        str(int(image_size)),
        "--style-size",
        str(int(style_size)),
        "--save-interval",
        str(int(save_interval)),
        "--skip-styles-with-epoch-at-least",
        str(int(target_epoch)),
        "--stdout-log",
        str(stdout_log),
        "--stderr-log",
        str(stderr_log),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep the local wikiarts5 SaMST lane advancing in 5-epoch common frontiers until convergence.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--stdout-log", type=Path, default=DEFAULT_STDOUT_LOG)
    parser.add_argument("--stderr-log", type=Path, default=DEFAULT_STDERR_LOG)
    parser.add_argument("--wsl-distro", default="f")
    parser.add_argument("--wsl-python", default="/root/venvs/samam/bin/python")
    parser.add_argument("--data-root", default="/mnt/f/wikiarts_5_full_notest")
    parser.add_argument("--styles", default="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--launch-cooldown-seconds", type=int, default=180)
    parser.add_argument("--convergence-json-key", default="converged")
    parser.add_argument("--epoch-interval", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--style-size", type=int, default=512)
    args = parser.parse_args()

    result_root = Path(args.result_root).expanduser().resolve()
    stdout_log = Path(args.stdout_log).expanduser().resolve()
    stderr_log = Path(args.stderr_log).expanduser().resolve()
    convergence_json = result_root / "eval_bundle" / "curve_convergence.json"
    result_root_wsl = _to_wsl_mount(result_root)
    train_match = f"run_samst_distinct5_local.py --data-root {args.data_root} --out-root {result_root_wsl}"
    last_launch_ts = 0.0

    while True:
        converged = _json_flag(convergence_json, key=str(args.convergence_json_key))
        train_busy = _wsl_has_process(distro=str(args.wsl_distro), match_text=train_match)
        common_epochs = _available_common_epochs(result_root, STYLE_NAMES)
        current_common = max(common_epochs) if common_epochs else 0
        next_target = max(int(args.epoch_interval), current_common + int(args.epoch_interval))
        payload = {
            "result_root": str(result_root),
            "converged": converged,
            "train_busy": train_busy,
            "current_common_epoch": current_common,
            "next_target_epoch": next_target,
            "convergence_json": str(convergence_json),
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if converged and (not train_busy):
            return 0
        if (not converged) and (not train_busy):
            if next_target > int(args.max_epochs):
                print(
                    json.dumps(
                        {
                            "action": "stop_no_more_epochs",
                            "current_common_epoch": current_common,
                            "max_epochs": int(args.max_epochs),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                return 0
            now = time.time()
            if now - last_launch_ts >= max(1, int(args.launch_cooldown_seconds)):
                _launch_segment(
                    result_root=result_root,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    wsl_distro=str(args.wsl_distro),
                    wsl_python=str(args.wsl_python),
                    data_root=str(args.data_root),
                    target_epoch=int(next_target),
                    save_interval=int(args.save_interval),
                    batch_size=int(args.batch_size),
                    image_size=int(args.image_size),
                    style_size=int(args.style_size),
                    styles=str(args.styles),
                )
                last_launch_ts = now
                print(
                    json.dumps(
                        {
                            "action": "launch_segment",
                            "target_epoch": int(next_target),
                            "result_root": str(result_root),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
