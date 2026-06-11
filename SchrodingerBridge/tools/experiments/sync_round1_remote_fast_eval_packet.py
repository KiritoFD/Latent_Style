from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from csv_utils import read_csv_rows
from round1_paths import round1_family_doc_dir
from round1_registry import ROUND1_FAMILY_SPECS


DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"
AUTO_START = "<!-- ROUND1_AUTO_STATUS:START -->"
AUTO_END = "<!-- ROUND1_AUTO_STATUS:END -->"

REMOTE_SCAN_PY = r"""
from pathlib import Path
import json

family_token = __FAMILY_TOKEN_JSON__
run_dir = Path(__import__('sys').argv[1])
eval_subdir = __import__('sys').argv[2]
root = run_dir / eval_subdir

payload = {
    "run_dir": str(run_dir),
    "eval_root": str(root),
    "epochs": [],
    "ckpts": [],
    "processes": {"train": [], "fast_eval": [], "posttrain_eval": []},
}
payload["ckpts"] = [p.name for p in sorted(run_dir.glob("epoch_*.pt"))]
if root.exists():
    for epoch_dir in sorted(root.glob("epoch_*")):
        payload["epochs"].append(
            {
                "epoch": epoch_dir.name,
                "has_metrics": (epoch_dir / "metrics.csv").is_file(),
                "has_summary": (epoch_dir / "summary.json").is_file(),
                "image_count": len(list((epoch_dir / "images").glob("*.png"))) if (epoch_dir / "images").is_dir() else 0,
            }
        )
for pid in Path("/proc").iterdir():
    if not pid.is_dir() or not pid.name.isdigit():
        continue
    try:
        raw = (pid / "cmdline").read_bytes()
    except Exception:
        continue
    if not raw:
        continue
    txt = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    if family_token not in txt:
        continue
    item = {"pid": pid.name, "cmd": txt}
    if (
        "watch_round1_family_fast_eval.py" in txt
        or "rerun_full_eval_for_run.py" in txt
        or "run_evaluation.py" in txt
        or "fast-eval.sh" in txt
        or "_fast_eval" in txt
        or "_fast-eval" in txt
    ):
        payload["processes"]["fast_eval"].append(item)
    elif "run_inmortal_posttrain_eval_when_done.py" in txt or "run_inmortal_posttrain_eval_latest_epochs_when_done.py" in txt:
        payload["processes"]["posttrain_eval"].append(item)
    elif "python3 - " in txt:
        continue
    else:
        payload["processes"]["train"].append(item)
print(json.dumps(payload, ensure_ascii=False))
"""


def _md_link(label: str, path: Path) -> str:
    return f"[{label}]({str(path).replace(chr(92), '/')})"


def _run(cmd: list[str], *, input_text: str | None = None, timeout_ms: int = 120000) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_ms / 1000,
        check=False,
    )


def _load_manifest_row(manifest_csv: Path, *, family_id: str) -> dict[str, str]:
    rows = read_csv_rows(manifest_csv)
    for row in rows:
        if str(row.get("family_id", "")).strip() == str(family_id).strip():
            return row
    raise KeyError(f"family_id not found in manifest: {family_id}")


def _remote_run_dir(*, row: dict[str, str], remote_workspace_root: str) -> str:
    run_dir = str(row.get("run_dir", "")).strip()
    if run_dir.startswith("./"):
        return f"{remote_workspace_root.rstrip('/')}/{run_dir[2:]}"
    return run_dir


def _family_patience(family_id: str) -> int:
    for spec in ROUND1_FAMILY_SPECS:
        if str(spec.family_id).strip() == str(family_id).strip():
            return int(spec.patience)
    return 4


def _read_curve(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _best(rows: list[dict[str, str]], *, style_key: str, lpips_key: str, mode: str) -> dict[str, str]:
    if mode == "style":
        return max(rows, key=lambda r: (float(r[style_key]), -float(r[lpips_key])))
    if mode == "lpips":
        return min(rows, key=lambda r: (float(r[lpips_key]), -float(r[style_key])))
    raise ValueError(mode)


def _latest(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda r: int("".join(ch for ch in str(r["epoch"]) if ch.isdigit()) or "-1"))


def _epoch_int(name: str) -> int:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else -1


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


def _upsert_auto_block(path: Path, body: str) -> None:
    block = f"{AUTO_START}\n{body.rstrip()}\n{AUTO_END}\n"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        start = text.find(AUTO_START)
        end = text.find(AUTO_END)
        if start >= 0 and end >= 0 and end >= start:
            before = text[:start].rstrip("\n")
            after = text[end + len(AUTO_END) :]
            after = re.sub(r"^(?:[ \t]*\n)+", "", after)
            new_text = before + "\n\n" + block
            if after:
                new_text += "\n" + after
        else:
            suffix = "" if text.endswith("\n") else "\n"
            new_text = text + suffix + "\n" + block
    else:
        new_text = block
    path.write_text(new_text, encoding="utf-8")


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _prune_incomplete_epoch_dirs(local_root: Path) -> None:
    for epoch_dir in local_root.glob("epoch_*"):
        if not epoch_dir.is_dir():
            continue
        if (epoch_dir / "summary.json").is_file():
            continue
        shutil.rmtree(epoch_dir, ignore_errors=True)


def _tracked_packet_advanced(
    previous_summary: dict[str, Any],
    *,
    local_curve_epochs: list[str],
    convergence: dict[str, Any],
) -> bool:
    if not previous_summary:
        return True
    previous_epochs = [str(x).strip() for x in (previous_summary.get("local_curve_epochs") or []) if str(x).strip()]
    if previous_epochs != local_curve_epochs:
        return True
    previous_convergence = previous_summary.get("convergence") or {}
    stable_keys = (
        "row_count",
        "newest_epoch",
        "last_pareto_epoch",
        "since_last_pareto",
        "best_transfer_clip_epoch",
        "best_transfer_lpips_epoch",
        "best_allpairs_clip_epoch",
        "best_allpairs_lpips_epoch",
        "converged",
    )
    for key in stable_keys:
        if previous_convergence.get(key) != convergence.get(key):
            return True
    return False


def _write_waiting_fast_eval_doc(
    *,
    docs_dir: Path,
    local_root: Path,
    summary_json: Path,
    run_name: str,
    remote_run_dir: str,
    eval_subdir: str,
    remote_scan: dict[str, Any],
    status: str = "waiting_for_first_remote_fast_eval_epoch",
) -> None:
    summary = {
        "status": str(status).strip(),
        "run_name": run_name,
        "remote_run_dir": remote_run_dir,
        "eval_subdir": eval_subdir,
        "remote_scan": remote_scan,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    auto_lines = [
        "## Auto Status",
        "",
        "- Authority root:",
        f"  - {_md_link(local_root.name, local_root)}",
        "- Remote fast-eval status:",
        f"  - `{str(status).strip()}`",
        f"- Run name:",
        f"  - `{run_name}`",
        f"- Remote run dir:",
        f"  - `{remote_run_dir}`",
        f"- Expected eval subdir:",
        f"  - `{eval_subdir}`",
    ]
    proc_groups = remote_scan.get("processes") or {}
    fast_eval = proc_groups.get("fast_eval") or []
    train = proc_groups.get("train") or []
    auto_lines.extend(
        [
            f"- Remote train pid count:",
            f"  - `{len(train)}`",
            f"- Remote fast-eval pid count:",
            f"  - `{len(fast_eval)}`",
            "- Sync summary:",
            f"  - {_md_link(summary_json.name, summary_json)}",
        ]
    )
    _upsert_auto_block(docs_dir / "fast_curve_read.md", "\n".join(auto_lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull a round-1 family's remote fast-eval packet and refresh local docs.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--host", default="administrator@100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--remote-workspace-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--eval-subdir", default="full_eval_fast_snapshot")
    parser.add_argument("--local-root", type=Path, default=None)
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    row = _load_manifest_row(manifest_csv, family_id=str(args.family_id))
    family_id = str(args.family_id).strip()
    run_name = str(row.get("run_name", "")).strip()
    remote_run_dir = _remote_run_dir(row=row, remote_workspace_root=str(args.remote_workspace_root))

    local_root = Path(args.local_root).expanduser() if args.local_root is not None else (SB_ROOT / "aaai2027" / f"round1_{family_id}_remote_full_eval_pull")
    if not local_root.is_absolute():
        local_root = (WORKSPACE / local_root).resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    docs_dir = round1_family_doc_dir(family_id=family_id, run_name=run_name)
    docs_dir.mkdir(parents=True, exist_ok=True)

    scan_cmd = [
        "ssh",
        "-p",
        str(int(args.port)),
        str(args.host),
        "wsl",
        "-d",
        str(args.wsl_distro),
        "python3",
        "-",
        str(remote_run_dir),
        str(args.eval_subdir),
    ]
    scan_proc = _run(
        scan_cmd,
        input_text=REMOTE_SCAN_PY.replace("__FAMILY_TOKEN_JSON__", json.dumps(run_name)),
        timeout_ms=30000,
    )
    if scan_proc.returncode != 0:
        raise RuntimeError(scan_proc.stderr or scan_proc.stdout or "remote scan failed")
    remote_scan = json.loads(scan_proc.stdout)

    summary_json = local_root / "sync_summary.json"
    if not bool(remote_scan.get("eval_root")) or not remote_scan.get("epochs"):
        _write_waiting_fast_eval_doc(
            docs_dir=docs_dir,
            local_root=local_root,
            summary_json=summary_json,
            run_name=run_name,
            remote_run_dir=remote_run_dir,
            eval_subdir=str(args.eval_subdir).strip(),
            remote_scan=remote_scan,
            status="waiting_for_first_remote_fast_eval_epoch",
        )
        print(docs_dir / "fast_curve_read.md")
        print(summary_json)
        return 0

    tar_name = f"{family_id}_{str(args.eval_subdir).strip()}.tar"
    pull_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "pull_remote_eval_dir.py"),
        "--host",
        str(args.host),
        "--port",
        str(int(args.port)),
        "--wsl-distro",
        str(args.wsl_distro),
        "--remote-dir",
        f"{remote_run_dir.rstrip('/')}/{str(args.eval_subdir).strip()}",
        "--local-dir",
        str(local_root),
        "--tar-name",
        tar_name,
    ]
    pull_proc = _run(pull_cmd, timeout_ms=120000)
    if pull_proc.returncode != 0:
        combined = (pull_proc.stderr or "") + (pull_proc.stdout or "")
        if "No such file or directory" in combined:
            _write_waiting_fast_eval_doc(
                docs_dir=docs_dir,
                local_root=local_root,
                summary_json=summary_json,
                run_name=run_name,
                remote_run_dir=remote_run_dir,
                eval_subdir=str(args.eval_subdir).strip(),
                remote_scan=remote_scan,
            )
            print(docs_dir / "fast_curve_read.md")
            print(summary_json)
            return 0
        raise RuntimeError(combined or "remote eval pull failed")

    _prune_incomplete_epoch_dirs(local_root)

    curve_csv = local_root / "clip_lpips_curve.csv"
    build_proc = _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_clip_lpips_curve_from_eval_root.py"),
            "--eval-root",
            str(local_root),
            "--output-csv",
            str(curve_csv),
        ],
        timeout_ms=120000,
    )
    if build_proc.returncode != 0:
        combined = (build_proc.stderr or "") + (build_proc.stdout or "")
        if "No summary.json files found" in combined:
            _write_waiting_fast_eval_doc(
                docs_dir=docs_dir,
                local_root=local_root,
                summary_json=summary_json,
                run_name=run_name,
                remote_run_dir=remote_run_dir,
                eval_subdir=str(args.eval_subdir).strip(),
                remote_scan=remote_scan,
                status="waiting_for_remote_fast_eval_summary",
            )
            print(docs_dir / "fast_curve_read.md")
            print(summary_json)
            return 0
        raise RuntimeError(combined or "curve rebuild failed")

    convergence_json = local_root / "round1_convergence.json"
    conv_proc = _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "report_round1_convergence.py"),
            "--curve-csv",
            str(curve_csv),
            "--patience",
            str(_family_patience(family_id)),
            "--output-json",
            str(convergence_json),
        ],
        timeout_ms=120000,
    )
    if conv_proc.returncode != 0:
        raise RuntimeError(conv_proc.stderr or conv_proc.stdout or "convergence rebuild failed")

    curve_rows = _read_curve(curve_csv)
    best_transfer_style = _best(curve_rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips", mode="style")
    best_transfer_lpips = _best(curve_rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips", mode="lpips")
    best_full_style = _best(curve_rows, style_key="full_clip_style", lpips_key="full_content_lpips", mode="style")
    latest = _latest(curve_rows)
    convergence = json.loads(convergence_json.read_text(encoding="utf-8"))
    remote_ckpts = [str(x) for x in (remote_scan.get("ckpts") or [])]
    remote_settled_epoch_names = [str(item.get("epoch", "")).strip() for item in remote_scan.get("epochs", []) if item.get("has_summary")]
    local_settled_epoch_names = [str(row.get("epoch", "")).strip() for row in curve_rows if str(row.get("epoch", "")).strip()]
    latest_remote_ckpt = max(remote_ckpts, key=_epoch_int) if remote_ckpts else ""
    latest_remote_settled_epoch = max(remote_settled_epoch_names, key=_epoch_int) if remote_settled_epoch_names else ""
    latest_local_settled_epoch = max(local_settled_epoch_names, key=_epoch_int) if local_settled_epoch_names else ""
    local_settled_epoch_set = set(local_settled_epoch_names)
    pending_ckpt_epochs = [name for name in remote_ckpts if name.replace(".pt", "") not in local_settled_epoch_set]

    summary = {
        "family_id": family_id,
        "run_name": run_name,
        "remote_run_dir": remote_run_dir,
        "eval_subdir": str(args.eval_subdir),
        "remote_scan": remote_scan,
        "curve_csv": str(curve_csv),
        "convergence_json": str(convergence_json),
        "best_transfer_style": best_transfer_style,
        "best_transfer_lpips": best_transfer_lpips,
        "best_full_style": best_full_style,
        "latest": latest,
        "convergence": convergence,
        "latest_remote_ckpt": latest_remote_ckpt,
        "remote_confirmed_settled_epochs": remote_settled_epoch_names,
        "latest_remote_settled_epoch": latest_remote_settled_epoch,
        "local_curve_epochs": local_settled_epoch_names,
        "latest_local_settled_epoch": latest_local_settled_epoch,
        "pending_ckpt_epochs": pending_ckpt_epochs,
    }
    previous_summary = _load_json_if_exists(summary_json)
    should_write_tracked = _tracked_packet_advanced(
        previous_summary,
        local_curve_epochs=local_settled_epoch_names,
        convergence=convergence,
    )
    if should_write_tracked:
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    settled_epochs = [item for item in remote_scan.get("epochs", []) if item.get("has_summary")]
    pending_epochs = [item for item in remote_scan.get("epochs", []) if item.get("has_metrics") and not item.get("has_summary")]
    auto_lines = [
        "## Auto Status",
        "",
        f"- Authority root:",
        f"  - {_md_link(local_root.name, local_root)}",
        f"- Pulled curve CSV:",
        f"  - {_md_link(curve_csv.name, curve_csv)}",
        f"- Pulled local curve rows:",
        f"  - `{len(curve_rows)}`",
        f"- Pulled local curve epochs:",
        f"  - `{', '.join(local_settled_epoch_names)}`",
    ]
    if latest_remote_ckpt:
        auto_lines.extend(
            [
                "- Latest remote checkpoint:",
                f"  - `{latest_remote_ckpt}`",
                "- Latest pulled local eval epoch:",
                f"  - `{latest_local_settled_epoch or 'none'}`",
            ]
        )
    if settled_epochs:
        auto_lines.extend(
            [
                "- Remote scan confirmed summary epochs:",
                f"  - `{', '.join(item['epoch'] for item in settled_epochs)}`",
                "- Latest remote confirmed eval epoch:",
                f"  - `{latest_remote_settled_epoch or 'none'}`",
            ]
        )
    if pending_ckpt_epochs:
        auto_lines.extend(
            [
                "- Remote checkpoints not yet pulled into local fast curve:",
                *[f"  - `{item}`" for item in pending_ckpt_epochs],
            ]
        )
    if pending_epochs:
        auto_lines.extend(
            [
                "- Pending remote epochs with metrics but no summary yet:",
                *[f"  - `{item['epoch']}`" for item in pending_epochs],
            ]
        )
    auto_lines.extend(
        [
            "- Best transfer `CLIP-S`:",
            f"  - `{best_transfer_style['epoch']}`",
            f"  - `style / lpips = {float(best_transfer_style['transfer_clip_style']):.4f} / {float(best_transfer_style['transfer_content_lpips']):.4f}`",
            "- Best transfer `LPIPS`:",
            f"  - `{best_transfer_lpips['epoch']}`",
            f"  - `style / lpips = {float(best_transfer_lpips['transfer_clip_style']):.4f} / {float(best_transfer_lpips['transfer_content_lpips']):.4f}`",
            "- Best all-pairs `CLIP-S`:",
            f"  - `{best_full_style['epoch']}`",
            f"  - `style / lpips = {float(best_full_style['full_clip_style']):.4f} / {float(best_full_style['full_content_lpips']):.4f}`",
            "- Latest settled point:",
            f"  - `{latest['epoch']}`",
            f"  - transfer `style / lpips = {float(latest['transfer_clip_style']):.4f} / {float(latest['transfer_content_lpips']):.4f}`",
            f"  - full `style / lpips = {float(latest['full_clip_style']):.4f} / {float(latest['full_content_lpips']):.4f}`",
            f"  - wall `= {float(latest['wall_total_seconds']):.2f}s`",
            "- Convergence snapshot:",
            f"  - `best_epoch = {convergence.get('best_epoch')}`",
            f"  - `since_last_pareto = {convergence.get('since_last_pareto')}`",
            f"  - `best_in_newest_2 = {convergence.get('best_in_newest_2')}`",
            f"  - `tail_flat = {convergence.get('tail_flat')}`",
            f"  - `closure_band = {_closure_band(convergence)}`",
            f"  - `criterion = {convergence.get('criterion')}`",
            f"  - `converged = {convergence.get('converged')}`",
            f"- Sync summary:",
            f"  - {_md_link(summary_json.name, summary_json)}",
        ]
    )

    fast_curve_doc = docs_dir / "fast_curve_read.md"
    if should_write_tracked:
        _upsert_auto_block(fast_curve_doc, "\n".join(auto_lines))

    print(fast_curve_doc)
    print(summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
