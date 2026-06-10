from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


AUTO_START = "<!-- WIKIARTS5_SAMST_AUTO_STATUS:START -->"
AUTO_END = "<!-- WIKIARTS5_SAMST_AUTO_STATUS:END -->"
RESULTS_ROOT = Path(r"G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results")
DEFAULT_DOC_PATH = Path(
    r"G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-10-wikiarts5-samst-repro.md"
)
STYLE_NAMES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]


def _md_link(label: str, path: Path) -> str:
    return f"[{label}]({str(path).replace(chr(92), '/')})"


def _default_result_root() -> Path:
    candidates = sorted(
        (path for path in RESULTS_ROOT.glob("samst_wikiarts5_wsl_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return RESULTS_ROOT / "samst_wikiarts5_wsl_missing"
    return candidates[-1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _upsert_auto_block(path: Path, body: str) -> None:
    block = f"{AUTO_START}\n{body.rstrip()}\n{AUTO_END}\n"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        start = text.find(AUTO_START)
        end = text.find(AUTO_END)
        if start >= 0 and end >= 0 and end >= start:
            new_text = text[:start] + block + text[end + len(AUTO_END) :]
        else:
            suffix = "" if text.endswith("\n") else "\n"
            new_text = text + suffix + "\n" + block
    else:
        new_text = block
    path.write_text(new_text, encoding="utf-8")


def _query_wsl_processes(*, distro: str, result_root: Path) -> list[dict[str, str]]:
    match_text = str(result_root).replace("\\", "/").replace("G:/", "/mnt/g/").replace("F:/", "/mnt/f/")
    cmd = (
        "ps -eo pid,etime,args | grep -F -- "
        + subprocess.list2cmdline([match_text])
        + " | grep -F 'run_samst_distinct5_local.py' | grep -v grep || true"
    )
    proc = subprocess.run(
        ["wsl", "-d", distro, "bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        match = re.match(r"^\s*(\d+)\s+(\S+)\s+(.*)$", text)
        if not match:
            continue
        pid, etime, command = match.groups()
        rows.append({"pid": pid, "etime": etime, "command": command})
    return rows


def _query_windows_processes(patterns: list[str]) -> list[dict[str, str]]:
    if not patterns:
        return []
    escaped = [p.replace("'", "''") for p in patterns]
    or_expr = " -or ".join([f"$_.CommandLine -match '{p}'" for p in escaped])
    command = (
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ {or_expr} }} | "
        "Select-Object ProcessId, Name, CommandLine | ConvertTo-Json -Depth 3"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    text = proc.stdout.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    rows = payload if isinstance(payload, list) else [payload]
    result: list[dict[str, str]] = []
    for row in rows:
        result.append(
            {
                "pid": str(row.get("ProcessId", "")),
                "name": str(row.get("Name", "")),
                "command": str(row.get("CommandLine", "")),
            }
        )
    return result


def _latest_train_log(logs_dir: Path) -> Path | None:
    candidates = sorted(logs_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _style_name_from_log_path(path: Path | None) -> str | None:
    if path is None:
        return None
    match = re.match(r"^train_(.+)\.log$", path.name)
    if not match:
        return None
    return match.group(1)


def _parse_latest_progress(log_text: str) -> dict[str, Any]:
    for line in log_text.splitlines():
        if "Epoch " not in line or "content:" not in line or "style:" not in line or "ae:" not in line or "total:" not in line:
            continue
        try:
            epoch_match = re.search(r"Epoch\s+(\d+):", line)
            step_match = re.search(r"\[(\d+)/(\d+)\]", line)
            content_match = re.search(r"content:\s*([0-9.]+)", line)
            style_match = re.search(r"style:\s*([0-9.]+)", line)
            ae_match = re.search(r"ae:\s*([0-9.]+)", line)
            total_match = re.search(r"total:\s*([0-9.]+)", line)
            if not all([epoch_match, step_match, content_match, style_match, ae_match, total_match]):
                continue
            last = {
                "epoch": int(epoch_match.group(1)),
                "step": int(step_match.group(1)),
                "step_total": int(step_match.group(2)),
                "content_loss": float(content_match.group(1)),
                "style_loss": float(style_match.group(1)),
                "ae_loss": float(ae_match.group(1)),
                "total_loss": float(total_match.group(1)),
            }
        except Exception:
            continue
    return last if 'last' in locals() else {}


def _gpu_sample() -> dict[str, str] | None:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    first = next((line.strip() for line in proc.stdout.splitlines() if line.strip()), "")
    if not first:
        return None
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return None
    return {
        "name": parts[0],
        "memory_used_mib": parts[1],
        "memory_total_mib": parts[2],
        "utilization_gpu_pct": parts[3],
    }


def _per_style_epoch_counts(result_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    ckpt_root = result_root / "checkpoints"
    for style in STYLE_NAMES:
        style_dir = ckpt_root / style
        counts[style] = len(list(style_dir.glob("epoch_*.model"))) if style_dir.is_dir() else 0
    return counts


def _common_saved_epochs(result_root: Path) -> list[int]:
    per_style: list[set[int]] = []
    ckpt_root = result_root / "checkpoints"
    for style in STYLE_NAMES:
        style_dir = ckpt_root / style
        epochs: set[int] = set()
        if style_dir.is_dir():
            for path in style_dir.glob("epoch_*.model"):
                match = re.search(r"epoch_(\d+)", path.stem)
                if match:
                    epochs.add(int(match.group(1)))
        per_style.append(epochs)
    if not per_style:
        return []
    return sorted(set.intersection(*per_style))


def _last_watch_event(result_root: Path) -> dict[str, Any]:
    watch_log = result_root / "watch_eval_every5.log"
    if not watch_log.is_file():
        return {}
    for line in reversed(watch_log.read_text(encoding="utf-8", errors="replace").splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the local wikiarts5 SaMST repro note from current logs and process state.")
    parser.add_argument("--result-root", type=Path, default=_default_result_root())
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--wsl-distro", default="f")
    args = parser.parse_args()

    result_root = Path(args.result_root).expanduser()
    doc_path = Path(args.doc_path).expanduser()
    if not result_root.is_absolute():
        result_root = (Path.cwd() / result_root).resolve()
    if not doc_path.is_absolute():
        doc_path = (Path.cwd() / doc_path).resolve()
    logs_dir = result_root / "logs"
    latest_log = _latest_train_log(logs_dir)
    active_style = _style_name_from_log_path(latest_log)
    latest_progress = _parse_latest_progress(_read_text(latest_log)) if latest_log is not None else {}
    processes = _query_wsl_processes(distro=str(args.wsl_distro), result_root=result_root)
    watcher_processes = _query_windows_processes(
        [
            "watch_wikiarts5_samst_eval_bundle.py",
            "watch_update_wikiarts5_samst_status.py",
            re.escape(str(result_root)),
        ]
    )
    gpu = _gpu_sample()
    epoch_counts = _per_style_epoch_counts(result_root)
    common_epochs = _common_saved_epochs(result_root)
    common_eval_epochs = [epoch for epoch in common_epochs if epoch > 0 and epoch % 5 == 0]
    last_watch_event = _last_watch_event(result_root)

    payload = {
        "result_root": str(result_root),
        "latest_log": "" if latest_log is None else str(latest_log),
        "active_style": "" if active_style is None else active_style,
        "latest_progress": latest_progress,
        "active_processes": processes,
        "watcher_processes": watcher_processes,
        "local_gpu": gpu,
        "per_style_epoch_counts": epoch_counts,
        "common_saved_epochs": common_epochs,
        "common_eval_epochs": common_eval_epochs,
        "last_watch_event": last_watch_event,
    }
    (result_root / "samst_live_status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "## Auto Status",
        "",
        f"- Result root: {_md_link(result_root.name, result_root)}",
        f"- Live JSON: {_md_link('samst_live_status.json', result_root / 'samst_live_status.json')}",
        f"- Active WSL process count: `{len(processes)}`",
    ]
    if processes:
        lines.append("- Active WSL process:")
        lines.append(f"  - `pid={processes[0]['pid']}` `etime={processes[0]['etime']}`")
    watcher_eval = next((row for row in watcher_processes if "watch_wikiarts5_samst_eval_bundle.py" in row["command"]), None)
    watcher_status = next((row for row in watcher_processes if "watch_update_wikiarts5_samst_status.py" in row["command"]), None)
    lines.append(f"- Eval watcher alive: `{'yes' if watcher_eval else 'no'}`")
    if watcher_eval:
        lines.append(f"  - `pid={watcher_eval['pid']}`")
    lines.append(f"- Status watcher alive: `{'yes' if watcher_status else 'no'}`")
    if watcher_status:
        lines.append(f"  - `pid={watcher_status['pid']}`")
    if latest_log is not None:
        lines.append(f"- Latest train log: {_md_link(latest_log.name, latest_log)}")
    if active_style is not None:
        lines.append(f"- Active style: `{active_style}`")
    if latest_progress:
        lines.extend(
            [
                "- Latest logged progress:",
                f"  - `epoch={latest_progress['epoch']}`",
                f"  - `step={latest_progress['step']} / {latest_progress['step_total']}`",
                f"  - `content/style/ae/total = {latest_progress['content_loss']:.2f} / {latest_progress['style_loss']:.2f} / {latest_progress['ae_loss']:.2f} / {latest_progress['total_loss']:.2f}`",
            ]
        )
    lines.extend(
        [
            "- Common saved epochs across all 5 styles:",
            f"  - `{', '.join(str(x) for x in common_epochs) if common_epochs else 'none yet'}`",
            "- Eligible every-5-epoch eval points currently present:",
            f"  - `{', '.join(str(x) for x in common_eval_epochs) if common_eval_epochs else 'none yet'}`",
        ]
    )
    lines.extend(
        [
            "- Per-style saved epoch checkpoints:",
            *[f"  - `{style}: {count}`" for style, count in epoch_counts.items()],
            "- First eval trigger condition:",
            "  - all five styles must each have `epoch_0005.model` before the every-5-epoch eval watcher launches the first full bundle",
            "- Important interpretation:",
            "  - the displayed `epoch` comes from the currently active single-style train log only",
            "  - this run trains styles sequentially, not all 5 styles in lockstep",
            "  - so `epoch=5` for `Early_Renaissance` still does not imply a common `epoch_0005.model` exists across all 5 style folders",
            "  - even for the current style, `epoch=5` means `the 5th epoch is in progress`; the `epoch_5.model` file is only written after that epoch finishes",
        ]
    )
    if last_watch_event:
        lines.extend(
            [
                "- Last eval-watch event:",
                f"  - `{json.dumps(last_watch_event, ensure_ascii=False)}`",
            ]
        )
    if gpu is not None:
        lines.extend(
            [
                "- Local GPU sample:",
                f"  - `{gpu['name']}`",
                f"  - `{gpu['memory_used_mib']} MiB / {gpu['memory_total_mib']} MiB`, `util={gpu['utilization_gpu_pct']}%`",
            ]
        )

    _upsert_auto_block(doc_path, "\n".join(lines))
    print(doc_path)
    print(result_root / "samst_live_status.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
