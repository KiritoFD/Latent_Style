from __future__ import annotations

import argparse
import csv
import json
import subprocess
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "experiments" / "2026-06-18-remote-real-run-audit"
DEFAULT_REMOTE_WORKTREE = "/mnt/i/Github/Latent_Style/SchrodingerBridge"
DEFAULT_STAGE_ROOTS = [
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_ot_rerun_lowrank_auto",
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_plain_path_distill_auto",
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_stage3_style_auto",
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical_auto",
]
JSON_SENTINEL = "__PHASE618_REMOTE_AUDIT_JSON__"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _ssh_run(*, host: str, port: int, user: str, remote_script: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            "ssh",
            "-p",
            str(port),
            "-T",
            "-o",
            "LogLevel=ERROR",
            f"{user}@{host}",
            "wsl -d Ubuntu-26.04 -- bash -s",
        ],
        input=remote_script.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _remote_audit_script(*, remote_worktree: str, stage_roots: list[str]) -> str:
    roots_json = json.dumps(stage_roots, ensure_ascii=False)
    return textwrap.dedent(
        f"""\
        set -euo pipefail
        cd {remote_worktree}
        python3 - <<'PY'
        import importlib.util
        import json
        from pathlib import Path

        ROOT = Path({remote_worktree!r})
        STAGE_ROOTS = json.loads({roots_json!r})
        JSON_SENTINEL = {JSON_SENTINEL!r}

        def load_module(path: Path, name: str):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"failed to load module from {{path}}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        BACKFILL = load_module(ROOT / "tools" / "experiments" / "backfill_phase618_stage_summary.py", "phase618_remote_backfill")
        AUDIT = load_module(ROOT / "tools" / "audit_phase618_run_validity.py", "phase618_remote_audit")
        AUTO = load_module(ROOT / "tools" / "experiments" / "phase616_auto.py", "phase618_remote_auto")

        def load_json(path: Path):
            return json.loads(path.read_text(encoding="utf-8"))

        def collect_run_entry(child: Path):
            curve_best = AUTO._best_curve_point(AUTO._resolve_artifact_run_dir(child), eval_subdir="full_eval_transfer")
            try:
                audit_full = AUDIT.audit_phase618_run_validity(run_dir=child) if (child / "config.json").is_file() else {{}}
                audit = AUTO._summarize_validity_result(audit_full, audit_path=Path("")) if audit_full else {{}}
            except Exception as exc:
                audit = {{"artifact_status": "failed", "error": repr(exc)}}
            return {{
                "name": child.name,
                "run_dir": str(child),
                "best_epoch": str((curve_best or {{}}).get("epoch", "") or ""),
                "best_epoch_int": int((curve_best or {{}}).get("epoch_int", 0) or 0),
                "style": (curve_best or {{}}).get("style", ""),
                "lpips": (curve_best or {{}}).get("lpips", ""),
                "gap": (curve_best or {{}}).get("gap", ""),
                "validity_audit": audit,
            }}

        payload = {{"remote_worktree": str(ROOT), "stage_roots": []}}
        for root_text in STAGE_ROOTS:
            root = Path(root_text)
            entry = {{
                "stage_root": str(root),
                "exists": root.is_dir(),
                "stage_summary_present": False,
                "stage_manifest_present": False,
                "backfill": {{}},
                "close_result_diagnosis": {{}},
                "best": {{}},
                "runs": [],
                "child_dirs": [],
            }}
            if not root.is_dir():
                payload["stage_roots"].append(entry)
                continue

            entry["child_dirs"] = sorted([child.name for child in root.iterdir() if child.is_dir()])
            stage_summary = root / "stage_summary.json"
            stage_manifest = root / "stage_manifest.json"
            entry["stage_summary_present"] = stage_summary.is_file()
            entry["stage_manifest_present"] = stage_manifest.is_file()

            if stage_summary.is_file():
                try:
                    entry["backfill"] = BACKFILL.backfill_stage_root(root)
                except Exception as exc:
                    entry["backfill"] = {{"status": "failed", "error": repr(exc)}}

            if stage_summary.is_file():
                summary = load_json(stage_summary)
                entry["close_result_diagnosis"] = dict(summary.get("close_result_diagnosis") or {{}})
                entry["best"] = dict(summary.get("best") or {{}})
                seen_run_dirs = set()
                seen_names = set()
                for run in summary.get("runs", []) or []:
                    if not isinstance(run, dict):
                        continue
                    run_dir_text = str(run.get("logical_run_dir") or run.get("run_dir") or "").strip()
                    run_dir = Path(run_dir_text) if run_dir_text else None
                    if run_dir_text:
                        seen_run_dirs.add(run_dir_text)
                    run_name = str(run.get("name", "") or "").strip()
                    if run_name:
                        seen_names.add(run_name)
                    audit = dict(run.get("validity_audit") or {{}})
                    if not audit and run_dir is not None and run_dir.is_dir():
                        try:
                            audit_full = AUDIT.audit_phase618_run_validity(run_dir=run_dir)
                            audit = AUTO._summarize_validity_result(audit_full, audit_path=Path(""))
                        except Exception as exc:
                            audit = {{"artifact_status": "failed", "error": repr(exc)}}
                    entry["runs"].append(
                        {{
                            "name": str(run.get("name", "") or ""),
                            "run_dir": run_dir_text,
                            "best_epoch": str(run.get("best_epoch", "") or ""),
                            "best_epoch_int": int(run.get("best_epoch_int", 0) or 0),
                            "style": run.get("best_transfer_clip_style", run.get("style", run.get("transfer_clip_style", ""))),
                            "lpips": run.get("best_transfer_content_lpips", run.get("lpips", run.get("transfer_content_lpips", ""))),
                            "gap": run.get("best_objective_gap", run.get("gap", run.get("objective_gap", ""))),
                            "validity_audit": audit,
                        }}
                    )
                for child in sorted(root.iterdir()):
                    if not child.is_dir() or child.name.startswith("_"):
                        continue
                    if str(child) in seen_run_dirs or child.name in seen_names:
                        continue
                    entry["runs"].append(collect_run_entry(child))
            else:
                for child in sorted(root.iterdir()):
                    if not child.is_dir() or child.name.startswith("_"):
                        continue
                    entry["runs"].append(collect_run_entry(child))
            payload["stage_roots"].append(entry)

        print(JSON_SENTINEL)
        print(json.dumps(payload, ensure_ascii=False))
        PY
        """
    )


def _parse_remote_payload(output: bytes) -> dict[str, Any]:
    text = output.decode("utf-8", errors="replace")
    idx = text.rfind(JSON_SENTINEL)
    if idx < 0:
        raise RuntimeError(f"remote audit output missing sentinel {JSON_SENTINEL!r}\n{text}")
    json_text = text[idx + len(JSON_SENTINEL) :].strip()
    return json.loads(json_text)


def _flatten_stage_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("stage_roots", []) or []:
        if not isinstance(item, dict):
            continue
        close = dict(item.get("close_result_diagnosis") or {})
        best = dict(item.get("best") or {})
        rows.append(
            {
                "stage_root": str(item.get("stage_root", "") or ""),
                "exists": bool(item.get("exists", False)),
                "stage_summary_present": bool(item.get("stage_summary_present", False)),
                "stage_manifest_present": bool(item.get("stage_manifest_present", False)),
                "run_count": len(item.get("runs", []) or []),
                "close_status": str(close.get("status", "") or ""),
                "close_interpretation": str(close.get("interpretation", "") or ""),
                "close_reason": str(close.get("reason", "") or ""),
                "best_name": str(best.get("name", "") or ""),
                "best_style": best.get("style", best.get("best_transfer_clip_style", "")),
                "best_lpips": best.get("lpips", best.get("best_transfer_content_lpips", "")),
                "best_gap": best.get("gap", best.get("best_objective_gap", "")),
                "backfill_status": str(dict(item.get("backfill") or {}).get("status", "") or ""),
                "child_dir_count": len(item.get("child_dirs", []) or []),
            }
        )
    return rows


def _flatten_run_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("stage_roots", []) or []:
        if not isinstance(item, dict):
            continue
        stage_root = str(item.get("stage_root", "") or "")
        for run in item.get("runs", []) or []:
            if not isinstance(run, dict):
                continue
            audit = dict(run.get("validity_audit") or {})
            rows.append(
                {
                    "stage_root": stage_root,
                    "name": str(run.get("name", "") or ""),
                    "run_dir": str(run.get("run_dir", "") or ""),
                    "best_epoch": str(run.get("best_epoch", "") or ""),
                    "best_epoch_int": int(run.get("best_epoch_int", 0) or 0),
                    "style": run.get("style", ""),
                    "lpips": run.get("lpips", ""),
                    "gap": run.get("gap", ""),
                    "artifact_status": str(audit.get("artifact_status", "") or ""),
                    "effect_contract": str(audit.get("effect_contract", "") or ""),
                    "suite": str(audit.get("suite", "") or ""),
                    "trust_level": str(audit.get("trust_level", "") or ""),
                    "scientific_reading": str(audit.get("scientific_reading", "") or ""),
                    "recommended_action": str(audit.get("recommended_action", "") or ""),
                    "issue_codes": ";".join(str(x) for x in (audit.get("issue_codes") or []) if str(x).strip()),
                }
            )
    return rows


def _build_readme(*, payload: dict[str, Any], stage_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Phase-618 Remote Real-Run Audit")
    lines.append("")
    lines.append("This artifact snapshots the current remote phase-618 stage roots through the same validity / close-result contract used locally.")
    lines.append("")
    lines.append("Generated by:")
    lines.append("")
    lines.append("```bash")
    lines.append("py -3.12 tools/experiments/build_phase618_remote_run_audit.py")
    lines.append("```")
    lines.append("")
    lines.append("| Stage root | Exists | stage_summary | Runs | Close status | Best |")
    lines.append("| --- | --- | --- | ---: | --- | --- |")
    for row in stage_rows:
        best_bits = ""
        if str(row.get("best_name", "") or "").strip():
            best_bits = (
                f"{row['best_name']} "
                f"({float(_f(row.get('best_style'), 0.0)):.4f} / {float(_f(row.get('best_lpips'), 0.0)):.4f})"
            )
        lines.append(
            f"| `{row['stage_root']}` | "
            f"{'yes' if row.get('exists') else 'no'} | "
            f"{'yes' if row.get('stage_summary_present') else 'no'} | "
            f"{int(row.get('run_count', 0) or 0)} | "
            f"{row.get('close_status', '') or 'n/a'} | "
            f"{best_bits or 'n/a'} |"
        )
    lines.append("")

    ot_row = next((row for row in stage_rows if str(row.get("stage_root", "")).endswith("20250618_ot_rerun_lowrank_auto")), None)
    plain_row = next((row for row in stage_rows if str(row.get("stage_root", "")).endswith("20250618_plain_path_distill_auto")), None)
    style_row = next((row for row in stage_rows if str(row.get("stage_root", "")).endswith("20250618_stage3_style_auto")), None)
    if ot_row is not None:
        lines.append("## Current read")
        lines.append("")
        lines.append(
            f"- remote OT rerun stage currently has `{int(ot_row.get('run_count', 0) or 0)}` discovered runs and "
            f"`close_result_diagnosis.status = {ot_row.get('close_status', 'n/a')!r}`."
        )
    if plain_row is not None:
        lines.append(
            f"- plain-path distill stage root exists: `{bool(plain_row.get('exists', False))}`; "
            f"stage-summary present: `{bool(plain_row.get('stage_summary_present', False))}`."
        )
    if style_row is not None:
        lines.append(
            f"- repaired style-sweep stage root exists: `{bool(style_row.get('exists', False))}`; "
            f"stage-summary present: `{bool(style_row.get('stage_summary_present', False))}`."
        )
    lines.append("")

    flagged_runs = [
        row
        for row in run_rows
        if str(row.get("artifact_status", "") or "").strip() in {"stale", "suspect", "confounded"}
    ]
    if flagged_runs:
        lines.append("## Flagged runs")
        lines.append("")
        lines.append("| Run | Stage | Audit | Notes |")
        lines.append("| --- | --- | --- | --- |")
        for row in flagged_runs:
            note = str(row.get("issue_codes", "") or "") or str(row.get("recommended_action", "") or "")
            lines.append(
                f"| {row.get('name', '')} | `{row.get('stage_root', '')}` | "
                f"{row.get('artifact_status', '')} | {note} |"
            )
        lines.append("")

    top_runs = sorted(
        [
            row
            for row in run_rows
            if str(row.get("name", "") or "").strip()
            and _f(row.get("gap"), 1e9) < 1.0
            and _f(row.get("style"), 0.0) > 0.0
            and _f(row.get("lpips"), 1.0) < 1.0
        ],
        key=lambda row: (
            _f(row.get("gap"), 1e9),
            -_f(row.get("style"), 0.0),
            _f(row.get("lpips"), 1.0),
            str(row.get("name", "") or ""),
        ),
    )[:8]
    lines.append("## Top discovered runs")
    lines.append("")
    lines.append("| Run | Stage | Style | LPIPS | Gap | Audit | Contract |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- |")
    for row in top_runs:
        lines.append(
            f"| {row.get('name', '')} | `{row.get('stage_root', '')}` | "
            f"{_f(row.get('style'), 0.0):.4f} | {_f(row.get('lpips'), 1.0):.4f} | {_f(row.get('gap'), 1e9):.4f} | "
            f"{row.get('artifact_status', '') or 'n/a'} | {row.get('effect_contract', '') or 'n/a'} |"
        )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `summary.json`")
    lines.append("- `remote_stage_audit.csv`")
    lines.append("- `remote_run_audit.csv`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill and audit current remote phase-618 stage roots into local JSON/CSV evidence.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--remote-worktree", default=DEFAULT_REMOTE_WORKTREE)
    parser.add_argument("--stage-root", action="append", default=[])
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    args = parser.parse_args()

    stage_roots = [str(x) for x in (args.stage_root or []) if str(x).strip()] or list(DEFAULT_STAGE_ROOTS)
    remote_script = _remote_audit_script(remote_worktree=str(args.remote_worktree), stage_roots=stage_roots)
    proc = _ssh_run(host=str(args.host), port=int(args.port), user=str(args.user), remote_script=remote_script)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    payload = _parse_remote_payload(proc.stdout)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_rows = _flatten_stage_rows(payload)
    run_rows = _flatten_run_rows(payload)
    _save_json(output_dir / "summary.json", payload)
    _save_csv(
        output_dir / "remote_stage_audit.csv",
        stage_rows,
        [
            "stage_root",
            "exists",
            "stage_summary_present",
            "stage_manifest_present",
            "run_count",
            "close_status",
            "close_interpretation",
            "close_reason",
            "best_name",
            "best_style",
            "best_lpips",
            "best_gap",
            "backfill_status",
            "child_dir_count",
        ],
    )
    _save_csv(
        output_dir / "remote_run_audit.csv",
        run_rows,
        [
            "stage_root",
            "name",
            "run_dir",
            "best_epoch",
            "best_epoch_int",
            "style",
            "lpips",
            "gap",
            "artifact_status",
            "effect_contract",
            "suite",
            "trust_level",
            "scientific_reading",
            "recommended_action",
            "issue_codes",
        ],
    )
    (output_dir / "README.md").write_text(
        _build_readme(payload=payload, stage_rows=stage_rows, run_rows=run_rows),
        encoding="utf-8",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
