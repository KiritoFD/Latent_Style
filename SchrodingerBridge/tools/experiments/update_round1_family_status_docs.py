from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_config
from round1_registry import COMMON_PARENT_CONFIG
from round1_paths import (
    infer_round1_family_id,
    round1_family_doc_dir,
    round1_fast_local_root,
    round1_localreview_root,
    round1_tokenizer_reconstruction_pretrain_config,
    round1_tokenizer_warmstart_config,
    round1_switch_smoke_artifact,
)


AUTO_START = "<!-- ROUND1_AUTO_STATUS:START -->"
AUTO_END = "<!-- ROUND1_AUTO_STATUS:END -->"
DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"
DEFAULT_MASTER = SB_ROOT / "docs" / "experiments" / "2026-06-10-round1-full-sweep-master.md"
DEFAULT_LOCAL_GPU_LOCK = SB_ROOT / "aaai2027" / ".local_gpu_eval.lock"
DEFAULT_REMOTE_HOST = "100.115.18.62"
DEFAULT_REMOTE_PORT = 2222
DEFAULT_REMOTE_USER = "administrator"
DEFAULT_REMOTE_WSL_DISTRO = "Ubuntu-26.04"
DEFAULT_REMOTE_WORKSPACE_ROOT = "/mnt/i/Github/Latent_Style"
DEFAULT_REMOTE_BAND_MIN_MIB = 9216
DEFAULT_REMOTE_BAND_MAX_MIB = 11059
DEFAULT_REMOTE_HARD_CAP_MIB = 11571

from csv_utils import manifest_fieldnames, read_csv_rows, write_csv_rows


def _read_rows(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path)


def _write_rows(path: Path, rows: list[dict[str, str]], *, fieldnames: list[str]) -> None:
    write_csv_rows(path, rows, fieldnames=fieldnames)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_optional(path: Path) -> dict | None:
    try:
        return _read_json(path)
    except Exception:
        return None


def _f(text: str | None) -> float | None:
    if text in (None, ""):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _epoch_int(epoch_name: str) -> int:
    digits = "".join(ch for ch in str(epoch_name) if ch.isdigit())
    return int(digits) if digits else -1


def _md_link(label: str, path: Path) -> str:
    return f"[{label}]({str(path.resolve()).replace(chr(92), '/')})"


def _upsert_auto_block(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    block = f"{AUTO_START}\n{body.rstrip()}\n{AUTO_END}\n"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        start = text.find(AUTO_START)
        end = text.find(AUTO_END)
        if start >= 0 and end >= 0 and end >= start:
            before = text[:start].rstrip("\n")
            after = re.sub(r"^(?:[ \t]*\n)+", "", text[end + len(AUTO_END) :])
            new_text = before + "\n\n" + block
            if after:
                new_text += "\n" + after
        else:
            suffix = "" if text.endswith("\n") else "\n"
            new_text = text + suffix + "\n" + block
    else:
        title = f"# {path.stem.replace('_', ' ').title()}\n\n"
        new_text = title + block
    path.write_text(new_text, encoding="utf-8")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _ssh_exec(*, host: str, port: int, user: str, remote_command: str) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "ssh",
            "-p",
            str(port),
            "-T",
            "-o",
            "LogLevel=ERROR",
            f"{user}@{host}",
            remote_command,
        ]
    )


def _infer_remote_train_log(*, run_dir: str, run_name: str, remote_workspace_root: str) -> str:
    run_dir_clean = str(run_dir or "").strip()
    if run_dir_clean.startswith("./"):
        run_dir_clean = run_dir_clean[2:]
    run_dir_clean = run_dir_clean.lstrip("/")
    if run_dir_clean:
        parent = Path(run_dir_clean).parent.as_posix()
        if parent == ".":
            parent = ""
        prefix = f"{remote_workspace_root.rstrip('/')}/{parent}".rstrip("/")
        return f"{prefix}/{run_name}_train.log"
    return f"{remote_workspace_root.rstrip('/')}/exp/inmortal-exp/{run_name}_train.log"


def _parse_remote_gpu_sample(text: str) -> dict[str, str] | None:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first:
        return None
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        return None
    return {
        "memory_used_mib": parts[0],
        "memory_total_mib": parts[1],
        "utilization_gpu_pct": parts[2],
    }


def _parse_remote_train_tail(text: str) -> dict[str, object]:
    payload: dict[str, object] = {"tail_text": text.strip()}
    epoch_match = re.findall(r"Epoch\s+(\d+)/(\d+):", text)
    if epoch_match:
        payload["epoch"] = int(epoch_match[-1][0])
        payload["epoch_total"] = int(epoch_match[-1][1])
    step_match = re.findall(r"\|\s+(\d+)/(\d+)\s+\[", text)
    if step_match:
        payload["step"] = int(step_match[-1][0])
        payload["step_total"] = int(step_match[-1][1])
    loss_match = re.findall(r"loss=([0-9.]+)", text)
    if loss_match:
        payload["loss"] = float(loss_match[-1])
    tswd_match = re.findall(r"tswd=([0-9.]+)", text)
    if tswd_match:
        payload["tswd"] = float(tswd_match[-1])
    return payload


def _classify_remote_vram_band(
    *,
    memory_used_mib: int | None,
    band_min_mib: int,
    band_max_mib: int,
    hard_cap_mib: int,
) -> str:
    if memory_used_mib is None:
        return "unknown"
    if memory_used_mib > int(hard_cap_mib):
        return "above_hard_cap"
    if memory_used_mib > int(band_max_mib):
        return "above_soft_band"
    if memory_used_mib < int(band_min_mib):
        return "under_band"
    return "in_band"


def _remote_process_scan_via_stdin(
    *,
    process_token: str,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
) -> dict[str, list[dict[str, str]]]:
    if not process_token:
        return {"train": [], "fast_eval": [], "posttrain_eval": []}
    scan_py = f"""
from pathlib import Path
import json

token = {process_token!r}
payload = {{"train": [], "fast_eval": [], "posttrain_eval": []}}
for pid in Path("/proc").iterdir():
    if not pid.is_dir() or not pid.name.isdigit():
        continue
    try:
        raw = (pid / "cmdline").read_bytes()
    except Exception:
        continue
    if not raw or token.encode() not in raw:
        continue
    cmd = raw.replace(b"\\x00", b" ").decode("utf-8", "replace").strip()
    item = {{"pid": pid.name, "cmd": cmd}}
    if (
        "watch_round1_family_fast_eval.py" in cmd
        or "run_evaluation.py" in cmd
        or "rerun_full_eval_for_run.py" in cmd
        or "fast-eval.sh" in cmd
        or "_fast_eval" in cmd
        or "_fast-eval" in cmd
    ):
        payload["fast_eval"].append(item)
    elif "run_inmortal_posttrain_eval" in cmd:
        payload["posttrain_eval"].append(item)
    elif (
        "SchrodingerBridge/src/run.py" in cmd
        or "src/run.py --config" in cmd
        or "/src/run.py --config" in cmd
    ):
        payload["train"].append(item)
print(json.dumps(payload, ensure_ascii=False))
"""
    proc = subprocess.run(
        [
            "ssh",
            "-p",
            str(port),
            "-T",
            "-o",
            "LogLevel=ERROR",
            f"{user}@{host}",
            "wsl",
            "-d",
            str(wsl_distro),
            "python3",
            "-",
        ],
        input=scan_py,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return {"train": [], "fast_eval": [], "posttrain_eval": []}
    try:
        payload = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {"train": [], "fast_eval": [], "posttrain_eval": []}
    if not isinstance(payload, dict):
        return {"train": [], "fast_eval": [], "posttrain_eval": []}
    return {
        "train": list(payload.get("train") or []),
        "fast_eval": list(payload.get("fast_eval") or []),
        "posttrain_eval": list(payload.get("posttrain_eval") or []),
    }


def _remote_runtime_snapshot(
    *,
    row: dict[str, str],
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_workspace_root: str,
    remote_log_lines: int,
) -> dict[str, object] | None:
    run_name = str(row.get("run_name", "")).strip()
    if not run_name:
        return None
    process_token = str(row.get("process_token", "")).strip() or run_name
    run_dir = str(row.get("run_dir", "")).strip()
    train_log = _infer_remote_train_log(run_dir=run_dir, run_name=run_name, remote_workspace_root=remote_workspace_root)
    processes = _remote_process_scan_via_stdin(
        process_token=process_token,
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
    )
    train_alive = bool(processes.get("train"))
    if not train_alive:
        return {
            "train_log": train_log,
            "gpu_sample": None,
            "tail": {},
            "gpu_returncode": None,
            "log_returncode": None,
            "processes": processes,
            "train_alive": False,
        }
    gpu_proc = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command="nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits",
    )
    log_proc = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=f"wsl -d {wsl_distro} --exec tail -n {int(remote_log_lines)} {train_log}",
    )
    gpu_sample = _parse_remote_gpu_sample(gpu_proc.stdout) if gpu_proc.returncode == 0 else None
    tail_info = _parse_remote_train_tail(log_proc.stdout) if log_proc.returncode == 0 else {"tail_text": ""}
    return {
        "train_log": train_log,
        "gpu_sample": gpu_sample,
        "tail": tail_info,
        "gpu_returncode": gpu_proc.returncode,
        "log_returncode": log_proc.returncode,
        "processes": processes,
        "train_alive": True,
    }


def _best_row(rows: list[dict[str, str]], *, style_key: str, lpips_key: str, prefer: str) -> dict[str, str] | None:
    best = None
    best_score = None
    for row in rows:
        style = _f(row.get(style_key))
        lpips = _f(row.get(lpips_key))
        if style is None or lpips is None:
            continue
        if prefer == "style":
            score = (style, -lpips)
        elif prefer == "lpips":
            score = (-lpips, style)
        else:
            score = (style, -lpips)
        if best_score is None or score > best_score:
            best = row
            best_score = score
    return best


def _latest_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: _epoch_int(str(row.get("epoch", ""))))


def _remote_fast_pull_root(*, family_id: str) -> Path:
    return SB_ROOT / "aaai2027" / f"round1_{str(family_id).strip()}_remote_full_eval_pull"


def _effective_fast_eval_paths(*, family_id: str, fast_root: Path, fast_eval_subdir: str) -> tuple[Path, Path, Path, dict | None]:
    remote_root = _remote_fast_pull_root(family_id=family_id)
    remote_curve = remote_root / "clip_lpips_curve.csv"
    remote_convergence = remote_root / "round1_convergence.json"
    remote_sync_summary = _read_json_optional(remote_root / "sync_summary.json")
    if remote_curve.is_file():
        return remote_root, remote_curve, remote_convergence, remote_sync_summary
    local_eval_root = fast_root / str(fast_eval_subdir).strip()
    return local_eval_root, local_eval_root / "clip_lpips_curve.csv", local_eval_root / "round1_convergence.json", remote_sync_summary


def _pending_checkpoints(checkpoint_root: Path, settled_epochs: set[str]) -> list[str]:
    names = []
    for path in checkpoint_root.glob("epoch_*.pt"):
        if path.suffix.lower() != ".pt" or ".pt." in path.name.lower():
            continue
        stem = path.stem
        if "." in stem:
            continue
        if stem not in settled_epochs:
            names.append(stem)
    return sorted(names, key=_epoch_int)


def _render_fast_curve_auto(
    *,
    family_id: str,
    fast_root: Path,
    curve_csv: Path,
    curve_rows: list[dict[str, str]],
    convergence: dict | None,
    remote_sync_summary: dict | None,
) -> str:
    remote_pending_epochs: list[str] = []
    latest_remote_ckpt = ""
    latest_remote_settled_epoch = ""
    latest_local_settled_epoch = ""
    pending_ckpt_epochs: list[str] = []
    remote_unconfirmed_local_settled_epochs: list[str] = []
    if isinstance(remote_sync_summary, dict):
        latest_remote_ckpt = str(remote_sync_summary.get("latest_remote_ckpt", "")).strip()
        latest_remote_settled_epoch = str(remote_sync_summary.get("latest_remote_settled_epoch", "")).strip()
        latest_local_settled_epoch = str(remote_sync_summary.get("latest_local_settled_epoch", "")).strip()
        pending_ckpt_epochs = [
            str(item).strip()
            for item in (remote_sync_summary.get("pending_ckpt_epochs") or [])
            if str(item).strip()
        ]
        remote_unconfirmed_local_settled_epochs = [
            str(item).strip()
            for item in (remote_sync_summary.get("remote_unconfirmed_local_settled_epochs") or [])
            if str(item).strip()
        ]
        explicit_pending = remote_sync_summary.get("remote_pending_metric_epochs")
        if isinstance(explicit_pending, list):
            remote_pending_epochs = [str(item).strip() for item in explicit_pending if str(item).strip()]
        else:
            remote_scan = remote_sync_summary.get("remote_scan") if isinstance(remote_sync_summary.get("remote_scan"), dict) else {}
            local_curve_epochs = {
                str(item).strip()
                for item in (remote_sync_summary.get("local_curve_epochs") or [])
                if str(item).strip()
            }
            for item in remote_scan.get("epochs", []) if isinstance(remote_scan.get("epochs"), list) else []:
                epoch = str((item or {}).get("epoch", "")).strip()
                if not epoch or epoch in local_curve_epochs:
                    continue
                has_metrics = bool((item or {}).get("has_metrics"))
                has_summary = bool((item or {}).get("has_summary"))
                if has_metrics and has_summary:
                    continue
                remote_pending_epochs.append(epoch)
    remote_pending_epochs = sorted(set(remote_pending_epochs), key=_epoch_int)

    if not curve_rows:
        if isinstance(remote_sync_summary, dict) and str(remote_sync_summary.get("status", "")).strip():
            remote_scan = remote_sync_summary.get("remote_scan") if isinstance(remote_sync_summary.get("remote_scan"), dict) else {}
            proc_groups = remote_scan.get("processes") if isinstance(remote_scan.get("processes"), dict) else {}
            train_count = len(proc_groups.get("train") or [])
            fast_eval_count = len(proc_groups.get("fast_eval") or [])
            return "\n".join(
                [
                    "## Auto Status",
                    "",
                    "- Authority root:",
                    f"  - {_md_link(fast_root.name, fast_root)}",
                    "- Remote fast-eval status:",
                    f"  - `{str(remote_sync_summary.get('status', '')).strip()}`",
                    "- Run name:",
                    f"  - `{str(remote_sync_summary.get('run_name', '')).strip()}`",
                    "- Remote run dir:",
                    f"  - `{str(remote_sync_summary.get('remote_run_dir', '')).strip()}`",
                    "- Expected eval subdir:",
                    f"  - `{str(remote_sync_summary.get('eval_subdir', '')).strip()}`",
                    "- Remote train pid count:",
                    f"  - `{train_count}`",
                    "- Remote fast-eval pid count:",
                    f"  - `{fast_eval_count}`",
                ]
            )
            if remote_pending_epochs:
                lines.extend(["- Remote pending eval epochs:", *[f"  - `{name}`" for name in remote_pending_epochs[-3:]]])
            return "\n".join(lines)
        return "\n".join(
            [
                "## Auto Status",
                "",
                "- No settled `clip_lpips_curve.csv` rows yet.",
                f"- Fast root: {_md_link(fast_root.name, fast_root)}",
            ]
        )
    best_transfer_style = _best_row(curve_rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips", prefer="style")
    best_transfer_lpips = _best_row(curve_rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips", prefer="lpips")
    best_full_style = _best_row(curve_rows, style_key="full_clip_style", lpips_key="full_content_lpips", prefer="style")
    latest = _latest_row(curve_rows)
    settled_epochs = {str(row.get("epoch", "")).strip() for row in curve_rows}
    pending = _pending_checkpoints(fast_root / "checkpoints", settled_epochs)
    lines = [
        "## Auto Status",
        "",
        f"- Fast root: {_md_link(fast_root.name, fast_root)}",
        f"- Curve CSV: {_md_link('clip_lpips_curve.csv', curve_csv)}",
    ]
    if latest_remote_ckpt:
        lines.extend(
            [
                "- Latest remote checkpoint:",
                f"  - `{latest_remote_ckpt}`",
            ]
        )
    if latest_local_settled_epoch:
        lines.extend(
            [
                "- Latest pulled local eval epoch:",
                f"  - `{latest_local_settled_epoch}`",
            ]
        )
    if latest_remote_settled_epoch:
        lines.extend(
            [
                "- Latest remote confirmed eval epoch:",
                f"  - `{latest_remote_settled_epoch}`",
            ]
        )
    if best_transfer_style:
        lines.extend(
            [
                "- Best transfer `CLIP-S`:",
                f"  - `{best_transfer_style['epoch']}`",
                f"  - `style / lpips = {float(best_transfer_style['transfer_clip_style']):.4f} / {float(best_transfer_style['transfer_content_lpips']):.4f}`",
            ]
        )
    if best_transfer_lpips:
        lines.extend(
            [
                "- Best transfer `LPIPS`:",
                f"  - `{best_transfer_lpips['epoch']}`",
                f"  - `style / lpips = {float(best_transfer_lpips['transfer_clip_style']):.4f} / {float(best_transfer_lpips['transfer_content_lpips']):.4f}`",
            ]
        )
    if best_full_style:
        lines.extend(
            [
                "- Best all-pairs `CLIP-S`:",
                f"  - `{best_full_style['epoch']}`",
                f"  - `style / lpips = {float(best_full_style['full_clip_style']):.4f} / {float(best_full_style['full_content_lpips']):.4f}`",
            ]
        )
    if latest:
        lines.extend(
            [
                "- Latest settled point:",
                f"  - `{latest['epoch']}`",
                f"  - transfer `style / lpips = {float(latest['transfer_clip_style']):.4f} / {float(latest['transfer_content_lpips']):.4f}`",
                f"  - full `style / lpips = {float(latest['full_clip_style']):.4f} / {float(latest['full_content_lpips']):.4f}`",
                f"  - wall `= {float(latest['wall_total_seconds']):.2f}s`",
            ]
        )
    if pending_ckpt_epochs:
        lines.extend(["- Remote checkpoints not yet pulled into local fast curve:", *[f"  - `{name}`" for name in pending_ckpt_epochs[-3:]]])
    if remote_pending_epochs:
        lines.extend(["- Remote pending eval epochs:", *[f"  - `{name}`" for name in remote_pending_epochs[-3:]]])
    if remote_unconfirmed_local_settled_epochs:
        lines.extend(
            [
                "- Local settled epochs waiting on remote summary confirmation:",
                *[f"  - `{name}`" for name in remote_unconfirmed_local_settled_epochs[-3:]],
            ]
        )
    if pending:
        lines.extend(["- Pending pulled checkpoints:", *[f"  - `{name}`" for name in pending[-3:]]])
    if convergence:
        since_field = "since_last_pareto" if convergence.get("since_last_pareto") is not None else "since_best"
        lines.extend(
            [
                "- Convergence snapshot:",
                f"  - `row_count = {convergence.get('row_count')}`",
                f"  - `best_epoch = {convergence.get('best_epoch')}`",
                f"  - `{since_field} = {convergence.get(since_field)}`",
                f"  - `best_in_newest_2 = {convergence.get('best_in_newest_2')}`",
                f"  - `tail_flat = {convergence.get('tail_flat')}`",
                f"  - `closure_band = {_closure_band(convergence)}`",
                f"  - `criterion = {convergence.get('criterion', 'transfer_only_best')}`",
                f"  - `converged = {convergence.get('converged')}`",
            ]
        )
    return "\n".join(lines)


def _render_local_review_auto(
    *,
    fast_root: Path,
    localreview_root: Path,
    fast_handoff_rows: list[dict[str, str]],
    handoff_rows: list[dict[str, str]],
    best_transfer_lpips_epoch: str | None,
    best_full_style_epoch: str | None,
    gpu_lock_owner: str,
) -> str:
    intro_csv = localreview_root / "full_eval_fresh_localreview_bestfew_introstyle.csv"
    dino_csv = localreview_root / "full_eval_fresh_localreview_bestfew_dino.csv"
    merged_csv = localreview_root / "full_eval_fresh_localreview_bestfew_introstyle_dino.csv"
    lines = [
        "## Auto Status",
        "",
        f"- Fast shortlist root: {_md_link(fast_root.name, fast_root)}",
        f"- Local review root: {_md_link(localreview_root.name, localreview_root)}",
    ]
    if fast_handoff_rows:
        lines.append("- Current canonical fast bestfew handoff:")
        lines.extend([f"  - `{str(row.get('reason', '')).strip()} = {str(row.get('epoch', '')).strip()}`" for row in fast_handoff_rows])
    else:
        lines.append("- No fast bestfew handoff CSV found yet.")
    if gpu_lock_owner:
        lines.append(f"- Current local GPU owner: `{gpu_lock_owner}`")
    if not handoff_rows:
        lines.append("- No localreview bestfew handoff CSV found yet.")
        return "\n".join(lines)
    lines.append("- Current localreview handoff:")
    reasons = []
    picked_epochs = set()
    for row in handoff_rows:
        reason = str(row.get("reason", "")).strip() or "unknown"
        epoch = str(row.get("epoch", "")).strip()
        picked_epochs.add(epoch)
        reasons.append((reason, epoch))
    lines.extend([f"  - `{reason} = {epoch}`" for reason, epoch in reasons])
    if best_transfer_lpips_epoch and best_transfer_lpips_epoch not in picked_epochs:
        lines.append(f"- Handoff is stale vs live fast curve: missing current best transfer-LPIPS `{best_transfer_lpips_epoch}`")
    if best_full_style_epoch and best_full_style_epoch not in picked_epochs:
        lines.append(f"- Handoff is stale vs live fast curve: missing current best all-pairs/style `{best_full_style_epoch}`")
    lines.extend(
        [
            "- Deep review artifacts:",
            f"  - `IntroStyle csv exists = {intro_csv.exists()}`",
            f"  - `DINO csv exists = {dino_csv.exists()}`",
            f"  - `Merged csv exists = {merged_csv.exists()}`",
        ]
    )
    return "\n".join(lines)


def _render_remote_run_auto(
    *,
    row: dict[str, str],
    fast_root: Path,
    localreview_root: Path,
    curve_rows: list[dict[str, str]],
    remote_runtime: dict[str, object] | None,
) -> str:
    run_name = str(row.get("run_name", "")).strip()
    current_status = str(row.get("decision_status", "")).strip().lower()
    pending = _pending_checkpoints(fast_root / "checkpoints", {str(x.get("epoch", "")).strip() for x in curve_rows})
    lines = [
        "## Auto Status",
        "",
        f"- Family id: `{row.get('family_id', '')}`",
        f"- Run name: `{run_name}`",
        f"- Remote run dir: `{row.get('run_dir', '')}`",
        f"- Config: {_md_link(Path(str(row.get('config_path', ''))).name, Path(str(row.get('config_path', ''))))}",
        f"- Manifest status: `{row.get('decision_status', '')}`",
        f"- Local fast root: {_md_link(fast_root.name, fast_root)}",
        f"- Local review root: {_md_link(localreview_root.name, localreview_root)}",
    ]
    smoke_status = str(row.get("switch_smoke_status", "")).strip()
    smoke_artifact = Path(str(row.get("switch_smoke_artifact", "")).strip()) if str(row.get("switch_smoke_artifact", "")).strip() else None
    smoke_row_count = str(row.get("switch_smoke_row_count", "")).strip()
    warmstart_config = Path(str(row.get("warmstart_config", "")).strip()) if str(row.get("warmstart_config", "")).strip() else None
    reconpretrain_config = Path(str(row.get("reconstruction_pretrain_config", "")).strip()) if str(row.get("reconstruction_pretrain_config", "")).strip() else None
    if smoke_status:
        lines.append(f"- Prelaunch switch smoke: `{smoke_status}`")
    if smoke_artifact is not None:
        lines.append(f"- Switch smoke artifact: {_md_link(smoke_artifact.name, smoke_artifact)}")
    if smoke_row_count:
        lines.append(f"- Switch smoke row count: `{smoke_row_count}`")
    if warmstart_config is not None:
        lines.append(f"- Tokenizer warmstart config: {_md_link(warmstart_config.name, warmstart_config)}")
    if reconpretrain_config is not None:
        lines.append(f"- Tokenizer reconstruction-pretrain config: {_md_link(reconpretrain_config.name, reconpretrain_config)}")
    wrote_live_runtime = False
    if remote_runtime:
        gpu_sample = remote_runtime.get("gpu_sample")
        tail = remote_runtime.get("tail") if isinstance(remote_runtime.get("tail"), dict) else {}
        train_log = str(remote_runtime.get("train_log", "")).strip()
        processes = remote_runtime.get("processes") if isinstance(remote_runtime.get("processes"), dict) else {}
        if gpu_sample and current_status == "running":
            band_status = _classify_remote_vram_band(
                memory_used_mib=int(str(gpu_sample.get("memory_used_mib", "0")).strip() or "0"),
                band_min_mib=DEFAULT_REMOTE_BAND_MIN_MIB,
                band_max_mib=DEFAULT_REMOTE_BAND_MAX_MIB,
                hard_cap_mib=DEFAULT_REMOTE_HARD_CAP_MIB,
            )
            lines.extend(
                [
                    "- Remote GPU live sample:",
                    f"  - `{gpu_sample['memory_used_mib']} MiB / {gpu_sample['memory_total_mib']} MiB`, `util={gpu_sample['utilization_gpu_pct']}%`",
                    f"  - `band_status={band_status}`",
                    f"  - `formal_status={'formal_in_band' if band_status == 'in_band' else f'nonformal_{band_status}'}`",
                ]
            )
            wrote_live_runtime = True
        if train_log:
            lines.append(f"- Remote train log: `{train_log}`")
        if not bool(remote_runtime.get("train_alive", True)):
            lines.append("- Remote train pid: not alive")
            if processes.get("fast_eval"):
                lines.append(f"- Remote fast-eval pid count: `{len(processes.get('fast_eval') or [])}`")
        epoch = tail.get("epoch")
        epoch_total = tail.get("epoch_total")
        step = tail.get("step")
        step_total = tail.get("step_total")
        loss = tail.get("loss")
        tswd = tail.get("tswd")
        if current_status == "running" and (epoch is not None or step is not None):
            lines.append("- Remote train progress:")
            if epoch is not None and epoch_total is not None:
                lines.append(f"  - `epoch {epoch}/{epoch_total}`")
            if step is not None and step_total is not None:
                lines.append(f"  - `step {step}/{step_total}`")
            if loss is not None:
                lines.append(f"  - `loss={float(loss):.4f}`")
            if tswd is not None:
                lines.append(f"  - `tswd={float(tswd):.4f}`")
            wrote_live_runtime = True
    if not wrote_live_runtime and current_status == "running":
        row_used = str(row.get("remote_live_memory_used_mib", "")).strip()
        row_total = str(row.get("remote_live_memory_total_mib", "")).strip()
        row_util = str(row.get("remote_live_util_pct", "")).strip()
        row_band = str(row.get("remote_live_band_status", "")).strip()
        row_formal = str(row.get("remote_live_formal_status", "")).strip()
        row_epoch = str(row.get("remote_live_epoch", "")).strip()
        row_epoch_total = str(row.get("remote_live_epoch_total", "")).strip()
        row_step = str(row.get("remote_live_step", "")).strip()
        row_step_total = str(row.get("remote_live_step_total", "")).strip()
        row_loss = str(row.get("remote_live_loss", "")).strip()
        row_tswd = str(row.get("remote_live_tswd", "")).strip()
        if row_used:
            lines.extend(
                [
                    "- Remote GPU live sample:",
                    f"  - `{row_used} MiB / {row_total or '?'} MiB`, `util={row_util or '?'}%`",
                    f"  - `band_status={row_band or 'unknown'}`",
                    f"  - `formal_status={row_formal or 'unknown'}`",
                ]
            )
        if row_epoch or row_step:
            lines.append("- Remote train progress:")
            if row_epoch and row_epoch_total:
                lines.append(f"  - `epoch {row_epoch}/{row_epoch_total}`")
            if row_step and row_step_total:
                lines.append(f"  - `step {row_step}/{row_step_total}`")
            if row_loss:
                lines.append(f"  - `loss={row_loss}`")
            if row_tswd:
                lines.append(f"  - `tswd={row_tswd}`")
    if pending:
        lines.extend(["- Pending local fast eval:", *[f"  - `{name}`" for name in pending[-3:]]])
    return "\n".join(lines)


def _render_master_auto(
    *,
    running_rows: list[dict[str, str]],
    family_row: dict[str, str],
    curve_rows: list[dict[str, str]],
    convergence: dict | None,
    remote_runtime: dict[str, object] | None,
) -> str:
    latest = _latest_row(curve_rows)
    best_transfer_style = _best_row(curve_rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips", prefer="style")
    best_transfer_lpips = _best_row(curve_rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips", prefer="lpips")
    best_full_style = _best_row(curve_rows, style_key="full_clip_style", lpips_key="full_content_lpips", prefer="style")
    lines = ["## Auto Active Status", ""]
    if running_rows:
        lines.append("- Running families:")
        lines.extend([f"  - `{row.get('family_id', '')}`" for row in running_rows])
    else:
        lines.append("- Running families: none")
    if running_rows:
        lines.extend(
            [
                f"- Active family: `{family_row.get('family_id', '')}`",
                f"- Decision status: `{family_row.get('decision_status', '')}`",
                f"- Batch / epochs / patience: `{family_row.get('batch_size', '')} / {family_row.get('num_epochs', '')} / {family_row.get('patience', '')}`",
            ]
        )
    else:
        lines.extend(
            [
                "- Active family: `none`",
                "- Decision status: `no_formal_running_lane`",
            ]
        )
    if remote_runtime and isinstance(remote_runtime.get("gpu_sample"), dict):
        gpu_sample = remote_runtime["gpu_sample"]
        band_status = _classify_remote_vram_band(
            memory_used_mib=int(str(gpu_sample.get("memory_used_mib", "0")).strip() or "0"),
            band_min_mib=DEFAULT_REMOTE_BAND_MIN_MIB,
            band_max_mib=DEFAULT_REMOTE_BAND_MAX_MIB,
            hard_cap_mib=DEFAULT_REMOTE_HARD_CAP_MIB,
        )
        lines.append(
            f"- Remote GPU live: `{gpu_sample['memory_used_mib']} / {gpu_sample['memory_total_mib']} MiB`, `util={gpu_sample['utilization_gpu_pct']}%`, `band={band_status}`"
        )
    elif remote_runtime and not bool(remote_runtime.get("train_alive", True)):
        row_used = str(family_row.get("remote_live_memory_used_mib", "")).strip()
        row_total = str(family_row.get("remote_live_memory_total_mib", "")).strip()
        row_util = str(family_row.get("remote_live_util_pct", "")).strip()
        row_band = str(family_row.get("remote_live_band_status", "")).strip()
        fast_eval_count = len((remote_runtime.get("processes") or {}).get("fast_eval") or [])
        if row_used:
            lines.append(
                f"- Remote GPU live (latest nonempty): `{row_used} / {row_total or '?'} MiB`, `util={row_util or '?'}%`, `band={row_band or 'unknown'}`"
            )
            lines.append(f"- Remote train pid: `not alive`; remote fast-eval pid count = `{fast_eval_count}`")
        else:
            lines.append(f"- Remote GPU live: `no active train pid; fast_eval_watchers={fast_eval_count}`")
    if best_transfer_style:
        lines.append(
            f"- Best transfer `CLIP-S`: `{best_transfer_style['epoch']}` -> `{float(best_transfer_style['transfer_clip_style']):.4f} / {float(best_transfer_style['transfer_content_lpips']):.4f}`"
        )
    if best_transfer_lpips:
        lines.append(
            f"- Best transfer `LPIPS`: `{best_transfer_lpips['epoch']}` -> `{float(best_transfer_lpips['transfer_clip_style']):.4f} / {float(best_transfer_lpips['transfer_content_lpips']):.4f}`"
        )
    if best_full_style:
        lines.append(
            f"- Best all-pairs `CLIP-S`: `{best_full_style['epoch']}` -> `{float(best_full_style['full_clip_style']):.4f} / {float(best_full_style['full_content_lpips']):.4f}`"
        )
    if latest:
        lines.append(
            f"- Latest settled fast point: `{latest['epoch']}` -> transfer `{float(latest['transfer_clip_style']):.4f} / {float(latest['transfer_content_lpips']):.4f}`"
        )
    if convergence:
        lines.append(
            f"- Convergence: `row_count={convergence.get('row_count')}, since_best={convergence.get('since_best')}, tail_flat={convergence.get('tail_flat')}, closure_band={_closure_band(convergence)}, converged={convergence.get('converged')}`"
        )
    return "\n".join(lines)


def _family_runtime_paths(row: dict[str, str]) -> tuple[str, Path, Path]:
    run_name = str(row.get("run_name", "")).strip()
    config_path = Path(str(row.get("config_path", "")))
    family_id = infer_round1_family_id(run_name=run_name, config_stem=config_path.stem) or str(row.get("family_id", "")).strip()
    return (
        family_id,
        round1_fast_local_root(family_id=family_id, run_name=run_name),
        round1_localreview_root(family_id=family_id, run_name=run_name),
    )


def _apply_runtime_sample_to_family_row(*, family_row: dict[str, str], sample: dict[str, object], band_min_mib: int, band_max_mib: int, hard_cap_mib: int) -> None:
    if not isinstance(sample, dict):
        return
    used_raw = str(sample.get("remote_live_memory_used_mib", "")).strip()
    total_raw = str(sample.get("remote_live_memory_total_mib", "")).strip()
    util_raw = str(sample.get("remote_live_util_pct", "")).strip()
    epoch_raw = str(sample.get("remote_live_epoch", "")).strip()
    epoch_total_raw = str(sample.get("remote_live_epoch_total", "")).strip()
    step_raw = str(sample.get("remote_live_step", "")).strip()
    step_total_raw = str(sample.get("remote_live_step_total", "")).strip()
    loss_raw = str(sample.get("remote_live_loss", "")).strip()
    tswd_raw = str(sample.get("remote_live_tswd", "")).strip()
    if not used_raw:
        return
    used_mib = int(float(used_raw))
    band_status = _classify_remote_vram_band(
        memory_used_mib=used_mib,
        band_min_mib=int(band_min_mib),
        band_max_mib=int(band_max_mib),
        hard_cap_mib=int(hard_cap_mib),
    )
    family_row["remote_live_memory_used_mib"] = used_raw
    family_row["remote_live_memory_total_mib"] = total_raw
    family_row["remote_live_util_pct"] = util_raw
    family_row["remote_live_band_status"] = band_status
    family_row["remote_live_formal_status"] = "formal_in_band" if band_status == "in_band" else f"nonformal_{band_status}"
    family_row["remote_live_epoch"] = epoch_raw
    family_row["remote_live_epoch_total"] = epoch_total_raw
    family_row["remote_live_step"] = step_raw
    family_row["remote_live_step_total"] = step_total_raw
    family_row["remote_live_loss"] = loss_raw
    family_row["remote_live_tswd"] = tswd_raw


def _manifest_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "family_id",
        "wave",
        "axis",
        "config_path",
        "run_name",
        "run_dir",
        "freeze_mode",
        "batch_size",
        "accumulation_steps",
        "num_epochs",
        "patience",
        "notes",
        "parent_config",
        "tokenizer_family",
        "backbone_attention_family",
        "solver_family",
        "semantic_supervision_family",
        "virtual_length_multiplier",
        "warmstart_config",
        "reconstruction_pretrain_config",
        "local_fast_root",
        "local_review_root",
        "switch_smoke_status",
        "switch_smoke_artifact",
        "switch_smoke_row_count",
        "best_ckpt",
        "best_transfer_lpips_ckpt",
        "best_allpairs_clip_style_ckpt",
        "latest_ckpt",
        "fast_converged",
        "convergence_reason",
        "decision_status",
        "remote_live_memory_used_mib",
        "remote_live_memory_total_mib",
        "remote_live_util_pct",
        "remote_live_band_status",
        "remote_live_formal_status",
        "remote_live_epoch",
        "remote_live_epoch_total",
        "remote_live_step",
        "remote_live_step_total",
        "remote_live_loss",
        "remote_live_tswd",
    ]
    all_fields = manifest_fieldnames(rows)
    seen = set(preferred)
    extras = [key for key in all_fields if key not in seen]
    return preferred + extras


def _latest_aggregate_smoke_summary(*, family_id: str) -> dict | None:
    smoke_files = sorted((SB_ROOT / "aaai2027").glob("round1_family_switch_smoke_*.json"))
    for path in reversed(smoke_files):
        payload = _read_json_optional(path)
        if not isinstance(payload, dict):
            continue
        for row in payload.get("results", []) if isinstance(payload.get("results"), list) else []:
            if str((row or {}).get("family_id", "")).strip() == str(family_id).strip():
                summary = dict(row)
                summary["_artifact_path"] = str(path)
                summary["_aggregate_row_count"] = str(payload.get("row_count", ""))
                return summary
    return None


def _switch_smoke_summary(*, family_id: str, run_name: str) -> tuple[dict | None, Path]:
    artifact = round1_switch_smoke_artifact(family_id=family_id, run_name=run_name)
    if artifact.exists():
        payload = _read_json_optional(artifact)
        if isinstance(payload, dict):
            if str(payload.get("family_id", "")).strip() == str(family_id).strip():
                summary = dict(payload)
                summary["_artifact_path"] = str(artifact)
                return summary, artifact
            if isinstance(payload.get("results"), list):
                for row in payload.get("results") or []:
                    if str((row or {}).get("family_id", "")).strip() == str(family_id).strip():
                        summary = dict(row)
                        summary["_artifact_path"] = str(artifact)
                        summary["_aggregate_row_count"] = str(payload.get("row_count", ""))
                        return summary, artifact
    aggregate = _latest_aggregate_smoke_summary(family_id=family_id)
    if isinstance(aggregate, dict) and str(aggregate.get("_artifact_path", "")).strip():
        return aggregate, Path(str(aggregate.get("_artifact_path", "")).strip())
    return aggregate, artifact


def _closure_band(convergence: dict | None) -> str:
    if not isinstance(convergence, dict):
        return "unknown"
    if bool(convergence.get("converged")):
        return "converged"
    patience = int(convergence.get("patience") or 0)
    since_last_pareto = convergence.get("since_last_pareto")
    since_best = convergence.get("since_best")
    best_in_newest_2 = bool(convergence.get("best_in_newest_2"))
    tail_flat = bool(convergence.get("tail_flat"))
    distance = None
    if since_last_pareto is not None:
        distance = int(since_last_pareto)
    elif since_best is not None:
        distance = int(since_best)
    if patience > 0 and distance is not None:
        if (not best_in_newest_2) and distance >= max(0, patience - 1):
            return "closure_ready" if tail_flat else "approaching_closure"
    return "open"


def _merge_family_row_into_manifest(path: Path, *, family_id: str, updated_row: dict[str, str]) -> None:
    latest_rows = _read_rows(path)
    merged = False
    for row in latest_rows:
        if str(row.get("family_id", "")).strip() != str(family_id).strip():
            continue
        for key, value in updated_row.items():
            row[key] = value
        merged = True
        break
    if not merged:
        latest_rows.append(dict(updated_row))
    _write_rows(path, latest_rows, fieldnames=_manifest_fieldnames(latest_rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh round-1 family status docs from manifest and machine-readable eval artifacts.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--master-note", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--fast-eval-subdir", default="full_eval_fast_snapshot")
    parser.add_argument("--review-eval-subdir", default="full_eval_fresh_localreview")
    parser.add_argument("--remote-live", action="store_true")
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    parser.add_argument("--remote-user", default=DEFAULT_REMOTE_USER)
    parser.add_argument("--remote-wsl-distro", default=DEFAULT_REMOTE_WSL_DISTRO)
    parser.add_argument("--remote-workspace-root", default=DEFAULT_REMOTE_WORKSPACE_ROOT)
    parser.add_argument("--remote-log-lines", type=int, default=60)
    parser.add_argument("--remote-band-min-mib", type=int, default=DEFAULT_REMOTE_BAND_MIN_MIB)
    parser.add_argument("--remote-band-max-mib", type=int, default=DEFAULT_REMOTE_BAND_MAX_MIB)
    parser.add_argument("--remote-hard-cap-mib", type=int, default=DEFAULT_REMOTE_HARD_CAP_MIB)
    args = parser.parse_args()

    manifest = Path(args.manifest_csv).resolve()
    rows = _read_rows(manifest)
    family_row = next((row for row in rows if str(row.get("family_id", "")).strip() == str(args.family_id).strip()), None)
    if family_row is None:
        raise RuntimeError(f"Family id not found in manifest: {args.family_id}")

    run_name = str(family_row.get("run_name", "")).strip()
    cfg = load_config(Path(str(family_row.get("config_path", ""))).resolve())
    family_id = infer_round1_family_id(run_name=run_name, config_stem=Path(str(family_row.get("config_path", ""))).stem) or str(args.family_id).strip()
    fast_root = round1_fast_local_root(family_id=family_id, run_name=run_name)
    localreview_root = round1_localreview_root(family_id=family_id, run_name=run_name)
    family_doc_dir = round1_family_doc_dir(family_id=family_id, run_name=run_name)
    switch_smoke, switch_smoke_artifact = _switch_smoke_summary(family_id=family_id, run_name=run_name)
    fast_eval_root, curve_csv, convergence_json, remote_sync_summary = _effective_fast_eval_paths(
        family_id=family_id,
        fast_root=fast_root,
        fast_eval_subdir=str(args.fast_eval_subdir).strip(),
    )
    curve_rows = _read_rows(curve_csv) if curve_csv.exists() else []
    convergence = _read_json(convergence_json) if convergence_json.exists() else None

    fast_handoff_csv = fast_root / f"{str(args.fast_eval_subdir).strip()}_bestfew_handoff.csv"
    fast_handoff_rows = _read_rows(fast_handoff_csv) if fast_handoff_csv.exists() else []
    handoff_csv = localreview_root / f"{str(args.review_eval_subdir).strip()}_bestfew_handoff.csv"
    handoff_rows = _read_rows(handoff_csv) if handoff_csv.exists() else []
    gpu_lock_payload = _read_json_optional(DEFAULT_LOCAL_GPU_LOCK)
    gpu_lock_owner = ""
    if isinstance(gpu_lock_payload, dict):
        gpu_lock_owner = str(gpu_lock_payload.get("owner", "")).strip()

    best_transfer_lpips = _best_row(curve_rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips", prefer="lpips")
    best_full_style = _best_row(curve_rows, style_key="full_clip_style", lpips_key="full_content_lpips", prefer="style")
    latest = _latest_row(curve_rows)
    remote_runtime = None
    if bool(args.remote_live):
        remote_runtime = _remote_runtime_snapshot(
            row=family_row,
            host=str(args.remote_host),
            port=int(args.remote_port),
            user=str(args.remote_user),
            wsl_distro=str(args.remote_wsl_distro),
            remote_workspace_root=str(args.remote_workspace_root),
            remote_log_lines=int(args.remote_log_lines),
        )
    runtime_summary_path = SB_ROOT / "aaai2027" / f"round1_{family_id}_runtime_summary.json"
    runtime_summary_payload = _read_json_optional(runtime_summary_path)

    family_row["parent_config"] = str(COMMON_PARENT_CONFIG)
    family_row["freeze_mode"] = str(((cfg.get("training") or {}).get("freeze_mode", family_row.get("freeze_mode", ""))))
    family_row["batch_size"] = str(((cfg.get("training") or {}).get("batch_size", family_row.get("batch_size", ""))))
    family_row["accumulation_steps"] = str(((cfg.get("training") or {}).get("accumulation_steps", family_row.get("accumulation_steps", ""))))
    family_row["num_epochs"] = str(((cfg.get("training") or {}).get("num_epochs", family_row.get("num_epochs", ""))))
    family_row["tokenizer_family"] = str(((cfg.get("model") or {}).get("tokenizer_family", "legacy_factorized")))
    family_row["backbone_attention_family"] = str(((cfg.get("model") or {}).get("backbone_attention_family", "legacy_semantic_crossattn")))
    family_row["solver_family"] = str(((cfg.get("model") or {}).get("solver_family", "euler_legacy")))
    family_row["semantic_supervision_family"] = str(((cfg.get("bridge") or {}).get("semantic_supervision_family", "legacy_terminal_swd")))
    family_row["virtual_length_multiplier"] = str(((cfg.get("data") or {}).get("virtual_length_multiplier", "")))
    warmstart_cfg = round1_tokenizer_warmstart_config(family_id=family_id)
    recon_cfg = round1_tokenizer_reconstruction_pretrain_config(family_id=family_id)
    family_row["warmstart_config"] = str(warmstart_cfg) if warmstart_cfg.exists() else ""
    family_row["reconstruction_pretrain_config"] = str(recon_cfg) if recon_cfg.exists() else ""
    family_row["local_fast_root"] = str(fast_root)
    family_row["local_review_root"] = str(localreview_root)
    family_row["switch_smoke_artifact"] = str(switch_smoke_artifact)
    if isinstance(switch_smoke, dict):
        family_row["switch_smoke_status"] = str(switch_smoke.get("status", "")).strip() or "unknown"
        smoke_row_count = ""
        if "results" in switch_smoke and isinstance(switch_smoke.get("results"), list):
            smoke_row_count = str(len(switch_smoke.get("results") or []))
        elif switch_smoke.get("_aggregate_row_count") is not None:
            smoke_row_count = str(switch_smoke.get("_aggregate_row_count", "")).strip()
        elif switch_smoke.get("row_count") is not None:
            smoke_row_count = str(switch_smoke.get("row_count", "")).strip()
        family_row["switch_smoke_row_count"] = smoke_row_count
    else:
        family_row["switch_smoke_status"] = ""
        family_row["switch_smoke_row_count"] = ""
    family_row["best_ckpt"] = "" if convergence is None else str(convergence.get("best_epoch", ""))
    family_row["best_transfer_lpips_ckpt"] = "" if best_transfer_lpips is None else str(best_transfer_lpips.get("epoch", ""))
    family_row["best_allpairs_clip_style_ckpt"] = "" if best_full_style is None else str(best_full_style.get("epoch", ""))
    family_row["latest_ckpt"] = "" if latest is None else str(latest.get("epoch", ""))
    family_row["fast_converged"] = "" if convergence is None else str(convergence.get("converged", ""))
    if convergence is None:
        family_row["convergence_reason"] = ""
    else:
        since_field = "since_last_pareto" if convergence.get("since_last_pareto") is not None else "since_best"
        family_row["convergence_reason"] = (
            f"best_in_newest_2={convergence.get('best_in_newest_2')}; "
            f"{since_field}={convergence.get(since_field)}; "
            f"tail_flat={convergence.get('tail_flat')}; "
            f"patience={convergence.get('patience')}"
        )
    if remote_runtime and isinstance(remote_runtime.get("gpu_sample"), dict):
        gpu_sample = remote_runtime["gpu_sample"]
        used_mib = int(str(gpu_sample.get("memory_used_mib", "0")).strip() or "0")
        band_status = _classify_remote_vram_band(
            memory_used_mib=used_mib,
            band_min_mib=int(args.remote_band_min_mib),
            band_max_mib=int(args.remote_band_max_mib),
            hard_cap_mib=int(args.remote_hard_cap_mib),
        )
        tail = remote_runtime.get("tail") if isinstance(remote_runtime.get("tail"), dict) else {}
        family_row["remote_live_memory_used_mib"] = str(used_mib)
        family_row["remote_live_memory_total_mib"] = str(gpu_sample.get("memory_total_mib", "")).strip()
        family_row["remote_live_util_pct"] = str(gpu_sample.get("utilization_gpu_pct", "")).strip()
        family_row["remote_live_band_status"] = band_status
        family_row["remote_live_formal_status"] = "formal_in_band" if band_status == "in_band" else f"nonformal_{band_status}"
        family_row["remote_live_epoch"] = "" if tail.get("epoch") is None else str(tail.get("epoch"))
        family_row["remote_live_epoch_total"] = "" if tail.get("epoch_total") is None else str(tail.get("epoch_total"))
        family_row["remote_live_step"] = "" if tail.get("step") is None else str(tail.get("step"))
        family_row["remote_live_step_total"] = "" if tail.get("step_total") is None else str(tail.get("step_total"))
        family_row["remote_live_loss"] = "" if tail.get("loss") is None else f"{float(tail.get('loss')):.4f}"
        family_row["remote_live_tswd"] = "" if tail.get("tswd") is None else f"{float(tail.get('tswd')):.4f}"
    elif remote_runtime and not bool(remote_runtime.get("train_alive", True)):
        family_row["remote_live_memory_used_mib"] = ""
        family_row["remote_live_memory_total_mib"] = ""
        family_row["remote_live_util_pct"] = ""
        family_row["remote_live_band_status"] = ""
        family_row["remote_live_formal_status"] = ""
        family_row["remote_live_epoch"] = ""
        family_row["remote_live_epoch_total"] = ""
        family_row["remote_live_step"] = ""
        family_row["remote_live_step_total"] = ""
        family_row["remote_live_loss"] = ""
        family_row["remote_live_tswd"] = ""
    elif isinstance(runtime_summary_payload, dict):
        latest_nonempty = runtime_summary_payload.get("latest_nonempty_sample") or runtime_summary_payload.get("latest_sample") or {}
        _apply_runtime_sample_to_family_row(
            family_row=family_row,
            sample=latest_nonempty if isinstance(latest_nonempty, dict) else {},
            band_min_mib=int(args.remote_band_min_mib),
            band_max_mib=int(args.remote_band_max_mib),
            hard_cap_mib=int(args.remote_hard_cap_mib),
        )
    if str(family_row.get("decision_status", "")).strip().lower() != "running":
        family_row["remote_live_memory_used_mib"] = ""
        family_row["remote_live_memory_total_mib"] = ""
        family_row["remote_live_util_pct"] = ""
        family_row["remote_live_band_status"] = ""
        family_row["remote_live_formal_status"] = ""
        family_row["remote_live_epoch"] = ""
        family_row["remote_live_epoch_total"] = ""
        family_row["remote_live_step"] = ""
        family_row["remote_live_step_total"] = ""
        family_row["remote_live_loss"] = ""
        family_row["remote_live_tswd"] = ""
    _merge_family_row_into_manifest(manifest, family_id=family_id, updated_row=family_row)
    rows = _read_rows(manifest)

    _upsert_auto_block(
        family_doc_dir / "fast_curve_read.md",
        _render_fast_curve_auto(
            family_id=family_id,
            fast_root=fast_eval_root,
            curve_csv=curve_csv,
            curve_rows=curve_rows,
            convergence=convergence,
            remote_sync_summary=remote_sync_summary,
        ),
    )
    _upsert_auto_block(
        family_doc_dir / "local_deep_review.md",
        _render_local_review_auto(
            fast_root=fast_root,
            localreview_root=localreview_root,
            fast_handoff_rows=fast_handoff_rows,
            handoff_rows=handoff_rows,
            best_transfer_lpips_epoch=None if best_transfer_lpips is None else str(best_transfer_lpips.get("epoch", "")),
            best_full_style_epoch=None if best_full_style is None else str(best_full_style.get("epoch", "")),
            gpu_lock_owner=gpu_lock_owner,
        ),
    )
    _upsert_auto_block(
        family_doc_dir / "remote_run.md",
        _render_remote_run_auto(
            row=family_row,
            fast_root=fast_root,
            localreview_root=localreview_root,
            curve_rows=curve_rows,
            remote_runtime=remote_runtime,
        ),
    )
    running_rows = [row for row in rows if str(row.get("decision_status", "")).strip().lower() == "running"]
    master_row = running_rows[0] if running_rows else family_row
    master_family_id, master_fast_root, _ = _family_runtime_paths(master_row)
    _, master_curve_csv, master_convergence_json, _ = _effective_fast_eval_paths(
        family_id=master_family_id,
        fast_root=master_fast_root,
        fast_eval_subdir=str(args.fast_eval_subdir).strip(),
    )
    master_curve_rows = _read_rows(master_curve_csv) if master_curve_csv.exists() else []
    master_convergence = _read_json(master_convergence_json) if master_convergence_json.exists() else None
    master_remote_runtime = None
    if bool(args.remote_live) and str(master_row.get("decision_status", "")).strip().lower() == "running":
        master_remote_runtime = _remote_runtime_snapshot(
            row=master_row,
            host=str(args.remote_host),
            port=int(args.remote_port),
            user=str(args.remote_user),
            wsl_distro=str(args.remote_wsl_distro),
            remote_workspace_root=str(args.remote_workspace_root),
            remote_log_lines=int(args.remote_log_lines),
        )
    _upsert_auto_block(
        Path(args.master_note).resolve(),
        _render_master_auto(
            running_rows=running_rows,
            family_row=master_row,
            curve_rows=master_curve_rows,
            convergence=master_convergence,
            remote_runtime=master_remote_runtime,
        ),
    )
    print(family_doc_dir)
    print(Path(args.master_note).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
