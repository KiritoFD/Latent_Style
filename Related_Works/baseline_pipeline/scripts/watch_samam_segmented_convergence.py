from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, Any], key: str) -> float:
    return float(str(row[key]))


def _i(row: dict[str, Any], key: str) -> int:
    return int(float(str(row[key])))


def _creates_new_pareto(rows: list[dict[str, str]], idx: int, *, style_key: str, lpips_key: str) -> bool:
    target_style = _f(rows[idx], style_key)
    target_lpips = _f(rows[idx], lpips_key)
    for prev in rows[:idx]:
        prev_style = _f(prev, style_key)
        prev_lpips = _f(prev, lpips_key)
        if prev_style >= target_style and prev_lpips <= target_lpips:
            if prev_style > target_style or prev_lpips < target_lpips:
                return False
    return True


def _convergence_payload(
    rows: list[dict[str, str]],
    *,
    patience: int,
    flat_eps_style: float,
    flat_eps_lpips: float,
    style_key: str,
    lpips_key: str,
) -> dict[str, Any]:
    best_idx = 0
    best_score = (_f(rows[0], style_key), -_f(rows[0], lpips_key))
    pareto_indices: list[int] = []
    for idx, row in enumerate(rows):
        score = (_f(row, style_key), -_f(row, lpips_key))
        if score > best_score:
            best_idx = idx
            best_score = score
        if _creates_new_pareto(rows, idx, style_key=style_key, lpips_key=lpips_key):
            pareto_indices.append(idx)

    newest_idx = len(rows) - 1
    best_in_newest_2 = best_idx >= max(0, newest_idx - 1)
    last_pareto_idx = pareto_indices[-1]
    since_last_pareto = newest_idx - last_pareto_idx
    tail = rows[max(0, newest_idx - 2) : newest_idx + 1]
    tail_style = [_f(row, style_key) for row in tail]
    tail_lpips = [_f(row, lpips_key) for row in tail]
    tail_flat = False
    if len(tail) >= 3:
        tail_flat = (
            max(tail_style) - min(tail_style) <= float(flat_eps_style)
            and max(tail_lpips) - min(tail_lpips) <= float(flat_eps_lpips)
        )

    converged = (not best_in_newest_2) and since_last_pareto >= int(patience) and tail_flat
    return {
        "row_count": len(rows),
        "best_step": _i(rows[best_idx], "step"),
        "best_clip_style": _f(rows[best_idx], style_key),
        "best_content_lpips": _f(rows[best_idx], lpips_key),
        "newest_step": _i(rows[newest_idx], "step"),
        "newest_clip_style": _f(rows[newest_idx], style_key),
        "newest_content_lpips": _f(rows[newest_idx], lpips_key),
        "best_in_newest_2": best_in_newest_2,
        "pareto_steps": [_i(rows[idx], "step") for idx in pareto_indices],
        "last_pareto_step": _i(rows[last_pareto_idx], "step"),
        "since_last_pareto": since_last_pareto,
        "tail_flat": tail_flat,
        "patience": int(patience),
        "style_key": str(style_key),
        "lpips_key": str(lpips_key),
        "converged": converged,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _to_wsl_mount(path: Path) -> str:
    text = str(path)
    if len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        remainder = text[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{remainder}" if remainder else f"/mnt/{drive}"
    return text.replace("\\", "/")


def _kill_matching_processes(*, wsl_distro: str, match_text: str) -> None:
    script = (
        "set -euo pipefail; "
        "mapfile -t lines < <(ps -eo pid,args | grep -F -- "
        + subprocess.list2cmdline([match_text])
        + " | grep -v grep || true); "
        "for line in \"${lines[@]}\"; do "
        "pid=$(echo \"$line\" | awk '{print $1}'); "
        "if [ -n \"$pid\" ]; then kill -TERM \"$pid\" 2>/dev/null || true; fi; "
        "done"
    )
    subprocess.run(
        ["wsl", "-d", wsl_distro, "bash", "-lc", script],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _run_post_command(command: list[str], *, cwd: Path | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        cwd=None if cwd is None else str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch a SaMAM segmented curve and stop the run when convergence is reached.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wsl-distro", default="f")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--flat-eps-style", type=float, default=0.006)
    parser.add_argument("--flat-eps-lpips", type=float, default=0.006)
    parser.add_argument("--style-key", default="clip_style")
    parser.add_argument("--lpips-key", default="content_lpips")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--kill-on-converged", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--refresh-cwd", type=Path, default=None)
    parser.add_argument(
        "--post-command",
        nargs=argparse.REMAINDER,
        default=None,
        help="Optional command to run after a new curve state is written. Use after `--post-command`, for example `--post-command python refresh.py`.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    curve_csv = root / "curve_metrics.csv"
    output_json = Path(args.output_json).expanduser() if args.output_json is not None else root / "curve_convergence.json"
    refresh_cwd = Path(args.refresh_cwd).expanduser() if args.refresh_cwd is not None else None
    post_command = list(args.post_command or [])
    last_refresh_key: tuple[Any, ...] | None = None
    cycles = 0
    while True:
        if curve_csv.is_file():
            rows = _read_rows(curve_csv)
            if rows:
                payload = _convergence_payload(
                    rows,
                    patience=max(1, int(args.patience)),
                    flat_eps_style=float(args.flat_eps_style),
                    flat_eps_lpips=float(args.flat_eps_lpips),
                    style_key=str(args.style_key),
                    lpips_key=str(args.lpips_key),
                )
                payload["curve_csv"] = str(curve_csv)
                _write_json(output_json, payload)
                print(json.dumps(payload, ensure_ascii=False), flush=True)
                refresh_key = (
                    payload.get("row_count"),
                    payload.get("best_step"),
                    payload.get("newest_step"),
                    payload.get("last_pareto_step"),
                    payload.get("converged"),
                )
                if post_command and refresh_key != last_refresh_key:
                    result = _run_post_command(post_command, cwd=refresh_cwd)
                    if result.stdout.strip():
                        print(result.stdout.rstrip(), flush=True)
                    if result.stderr.strip():
                        print(result.stderr.rstrip(), file=sys.stderr, flush=True)
                    if result.returncode == 0:
                        last_refresh_key = refresh_key
                    else:
                        print(
                            f"post_command_failed returncode={result.returncode} command={post_command!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                if payload["converged"] and bool(args.kill_on_converged):
                    _kill_matching_processes(wsl_distro=str(args.wsl_distro), match_text=_to_wsl_mount(root))
                    return 0
        cycles += 1
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
