from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


AUTO_START = "<!-- WIKIARTS5_BASELINE_AUTO_STATUS:START -->"
AUTO_END = "<!-- WIKIARTS5_BASELINE_AUTO_STATUS:END -->"
PAGE1_AUTO_START = "<!-- WIKIARTS5_PAGE1_AUTO_STATUS:START -->"
PAGE1_AUTO_END = "<!-- WIKIARTS5_PAGE1_AUTO_STATUS:END -->"
DEFAULT_RESULT_ROOT = Path(
    r"G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samam_wikiarts5_patch8_segmented_20260610_094447"
)
DEFAULT_DOC_PATH = Path(
    r"G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-10-wikiarts5-baseline-repro.md"
)
DEFAULT_PAGE1_DOC_PATH = Path(
    r"G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-10-wikiarts5-page1-read.md"
)
DEFAULT_PAGE1_SCRIPT_PATH = Path(
    r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\scripts_gen_wikiarts5_page1_assets.py"
)
DEFAULT_PAGE1_SUMMARY_JSON = Path(
    r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\wikiarts5_page1\wikiarts5_page1_summary.json"
)
DEFAULT_PAGE1_SUMMARY_CSV = Path(
    r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\wikiarts5_page1\wikiarts5_page1_summary.csv"
)
DEFAULT_PAGE1_CURVE_CSV = Path(
    r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\wikiarts5_page1\wikiarts5_page1_curve.csv"
)
DEFAULT_PAGE1_SUMMARY_PNG = Path(
    r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\figures\fig_wikiarts5_page1_summary.png"
)
DEFAULT_PAGE1_QUAL_PNG = Path(
    r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\figures\fig_wikiarts5_qualitative_main.png"
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(row: dict[str, Any], key: str) -> float:
    return float(str(row[key]))


def _i(row: dict[str, Any], key: str) -> int:
    return int(float(str(row[key])))


def _md_link(label: str, path: Path) -> str:
    target = path if path.is_absolute() else Path.cwd() / path
    return f"[{label}]({str(target).replace(chr(92), '/')})"


def _safe_float(value: Any) -> float:
    return float(str(value))


def _to_wsl_mount(path: Path) -> str:
    text = str(path)
    if len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        remainder = text[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{remainder}" if remainder else f"/mnt/{drive}"
    return text.replace("\\", "/")


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


def _compute_convergence(
    rows: list[dict[str, str]],
    *,
    patience: int = 4,
    flat_eps_style: float = 0.006,
    flat_eps_lpips: float = 0.006,
    style_key: str = "clip_style",
    lpips_key: str = "content_lpips",
) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "best_step": None,
            "last_pareto_step": None,
            "since_last_pareto": None,
            "tail_flat": False,
            "converged": False,
        }

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
    last_pareto_idx = pareto_indices[-1]
    since_last_pareto = newest_idx - last_pareto_idx
    best_in_newest_2 = best_idx >= max(0, newest_idx - 1)
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


def _best_row(rows: list[dict[str, str]], *, prefer: str) -> dict[str, str] | None:
    best = None
    best_score = None
    for row in rows:
        clip_style = _f(row, "clip_style")
        lpips = _f(row, "content_lpips")
        if prefer == "clip":
            score = (clip_style, -lpips)
        elif prefer == "lpips":
            score = (-lpips, clip_style)
        else:
            raise ValueError(f"Unknown preference: {prefer}")
        if best_score is None or score > best_score:
            best = row
            best_score = score
    return best


def _best_row_by_keys(rows: list[dict[str, str]], *, primary_key: str, secondary_key: str, maximize_primary: bool) -> dict[str, str] | None:
    best = None
    best_score = None
    for row in rows:
        primary = _f(row, primary_key)
        secondary = _f(row, secondary_key)
        score = (primary, -secondary) if maximize_primary else (-primary, secondary)
        if best_score is None or score > best_score:
            best = row
            best_score = score
    return best


def _latest_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: _i(row, "step"))


def _run_local_gpu_query() -> dict[str, str] | None:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        return None
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


def _query_wsl_processes(*, distro: str, result_root: Path) -> list[dict[str, str]]:
    mount_root = _to_wsl_mount(result_root)
    cmd = (
        "ps -eo pid,etime,args | grep -F -- "
        + subprocess.list2cmdline([mount_root])
        + " | grep -F 'train_SaMam.py' | grep -v grep || true"
    )
    proc = subprocess.run(
        ["wsl", "-d", distro, "bash", "-lc", cmd],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
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
        target = ""
        target_match = re.search(r"--iterations\s+(\d+)", command)
        if target_match:
            target = target_match.group(1)
        rows.append({"pid": pid, "etime": etime, "command": command, "target_step": target})
    return rows


def _upsert_marked_block(path: Path, body: str, *, start_marker: str, end_marker: str) -> None:
    block = f"{start_marker}\n{body.rstrip()}\n{end_marker}\n"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        start = text.find(start_marker)
        end = text.find(end_marker)
        if start >= 0 and end >= 0 and end >= start:
            new_text = text[:start] + block + text[end + len(end_marker) :]
        else:
            suffix = "" if text.endswith("\n") else "\n"
            new_text = text + suffix + "\n" + block
    else:
        new_text = block
    path.write_text(new_text, encoding="utf-8")


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
        new_text = "# WikiArts-5 Baseline Repro\n\n" + block
    path.write_text(new_text, encoding="utf-8")


def _run_python_script(script_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", str(script_path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def _render_auto_status(
    *,
    result_root: Path,
    curve_rows: list[dict[str, str]],
    convergence: dict[str, Any] | None,
    gpu: dict[str, str] | None,
    processes: list[dict[str, str]],
) -> str:
    latest = _latest_row(curve_rows)
    best_clip = _best_row(curve_rows, prefer="clip")
    best_lpips = _best_row(curve_rows, prefer="lpips")
    has_transfer = bool(curve_rows) and ("transfer_clip_style" in curve_rows[0]) and ("transfer_lpips" in curve_rows[0])
    transfer_best_clip = _best_row_by_keys(curve_rows, primary_key="transfer_clip_style", secondary_key="transfer_lpips", maximize_primary=True) if has_transfer else None
    transfer_best_lpips = _best_row_by_keys(curve_rows, primary_key="transfer_lpips", secondary_key="transfer_clip_style", maximize_primary=False) if has_transfer else None
    active_target = latest["step"] if latest is not None else ""
    if processes:
        for proc in processes:
            if proc.get("target_step"):
                active_target = proc["target_step"]
                break
    current_segment = ""
    if latest is not None and active_target:
        latest_step = _i(latest, "step")
        target_step = int(active_target)
        if target_step > latest_step:
            current_segment = f"`{latest_step} -> {target_step}`"
        else:
            current_segment = f"`{target_step}`"

    lines = [
        "## Auto Status",
        "",
        f"- Result root: {_md_link(result_root.name, result_root)}",
        f"- Curve CSV: {_md_link('curve_metrics.csv', result_root / 'curve_metrics.csv')}",
        f"- Convergence JSON: {_md_link('curve_convergence.json', result_root / 'curve_convergence.json')}",
        f"- CLIP/LPIPS curve: {_md_link('clip_lpips_curve.png', result_root / 'clip_lpips_curve.png')}",
        f"- Timing curve: {_md_link('timing_curve.png', result_root / 'timing_curve.png')}",
        f"- Active WSL training process count: `{len(processes)}`",
    ]
    if processes:
        lines.append("- Active WSL process:")
        for proc in processes[:1]:
            lines.append(
                f"  - `pid={proc['pid']}` `etime={proc['etime']}` `target_step={proc.get('target_step', '') or 'unknown'}`"
            )
    if current_segment:
        lines.append(f"- Current training segment: {current_segment}")
    if latest is not None:
        lines.extend(
            [
                "- Latest settled point:",
                f"  - `step={_i(latest, 'step')}`",
                f"  - `clip_style / lpips / clip_content = {_f(latest, 'clip_style'):.4f} / {_f(latest, 'content_lpips'):.4f} / {_f(latest, 'clip_content'):.4f}`",
                f"  - `infer_wall_seconds / metric_wall_seconds = {_f(latest, 'infer_wall_seconds'):.2f} / {_f(latest, 'metric_wall_seconds'):.2f}`",
            ]
        )
    if best_clip is not None:
        lines.extend(
            [
                "- Best `CLIP-S`:",
                f"  - `step={_i(best_clip, 'step')}`",
                f"  - `clip_style / lpips = {_f(best_clip, 'clip_style'):.4f} / {_f(best_clip, 'content_lpips'):.4f}`",
            ]
        )
    if best_lpips is not None:
        lines.extend(
            [
                "- Best `LPIPS`:",
                f"  - `step={_i(best_lpips, 'step')}`",
                f"  - `clip_style / lpips = {_f(best_lpips, 'clip_style'):.4f} / {_f(best_lpips, 'content_lpips'):.4f}`",
            ]
        )
    if has_transfer and transfer_best_clip is not None:
        lines.extend(
            [
                "- Best transfer `CLIP-S`:",
                f"  - `step={_i(transfer_best_clip, 'step')}`",
                f"  - `clip_style / lpips = {_f(transfer_best_clip, 'transfer_clip_style'):.4f} / {_f(transfer_best_clip, 'transfer_lpips'):.4f}`",
            ]
        )
    if has_transfer and transfer_best_lpips is not None:
        lines.extend(
            [
                "- Best transfer `LPIPS`:",
                f"  - `step={_i(transfer_best_lpips, 'step')}`",
                f"  - `clip_style / lpips = {_f(transfer_best_lpips, 'transfer_clip_style'):.4f} / {_f(transfer_best_lpips, 'transfer_lpips'):.4f}`",
            ]
        )
    if has_transfer and latest is not None:
        lines.extend(
            [
                "- Latest transfer point:",
                f"  - `step={_i(latest, 'step')}`",
                f"  - `clip_style / lpips = {_f(latest, 'transfer_clip_style'):.4f} / {_f(latest, 'transfer_lpips'):.4f}`",
            ]
        )
    if convergence is not None:
        lines.extend(
            [
                "- Convergence snapshot:",
                f"  - `row_count = {convergence.get('row_count')}`",
                f"  - `best_step = {convergence.get('best_step')}`",
                f"  - `last_pareto_step = {convergence.get('last_pareto_step')}`",
                f"  - `since_last_pareto = {convergence.get('since_last_pareto')}`",
                f"  - `tail_flat = {convergence.get('tail_flat')}`",
                f"  - `style_key / lpips_key = {convergence.get('style_key', 'clip_style')} / {convergence.get('lpips_key', 'content_lpips')}`",
                f"  - `converged = {convergence.get('converged')}`",
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
    return "\n".join(lines)


def _render_page1_auto_status(
    *,
    summary: dict[str, Any],
    summary_json_path: Path,
    summary_csv_path: Path,
    curve_csv_path: Path,
    summary_png_path: Path,
    qual_png_path: Path,
) -> str:
    idt = dict(summary.get("idt", {}))
    old = dict(summary.get("samam_2250", {}))
    best_clip = dict(summary.get("wikiarts5_best_clip", {}))
    best_lpips = dict(summary.get("wikiarts5_best_lpips", {}))
    latest = dict(summary.get("wikiarts5_latest", {}))
    lines = [
        "## Auto Status",
        "",
        f"- Fixed held-out test split: `{summary.get('fixed_test_split', '')}`",
        f"- Summary JSON: {_md_link(summary_json_path.name, summary_json_path)}",
        f"- Summary CSV: {_md_link(summary_csv_path.name, summary_csv_path)}",
        f"- Curve CSV: {_md_link(curve_csv_path.name, curve_csv_path)}",
        f"- Summary figure: {_md_link(summary_png_path.name, summary_png_path)}",
        f"- Qualitative figure: {_md_link(qual_png_path.name, qual_png_path)}",
        "- `IDT` floor:",
        f"  - `transfer CLIP-S = {_safe_float(idt.get('transfer_clip_style', 0.0)):.4f}`",
        f"  - `all-pairs CLIP-S = {_safe_float(idt.get('all_pairs_clip_style', 0.0)):.4f}`",
        "- Old `SaMAM-2250`:",
        f"  - `transfer CLIP-S / LPIPS = {_safe_float(old.get('transfer_clip_style', 0.0)):.4f} / {_safe_float(old.get('transfer_lpips', 0.0)):.4f}`",
        f"  - `delta_idt_transfer = {_safe_float(old.get('delta_idt_transfer', 0.0)):.4f}`",
        "- New `wikiarts5` best transfer-`CLIP-S`:",
        f"  - `step = {best_clip.get('step', '')}`",
        f"  - `transfer CLIP-S / LPIPS = {_safe_float(best_clip.get('transfer_clip_style', 0.0)):.4f} / {_safe_float(best_clip.get('transfer_lpips', 0.0)):.4f}`",
        f"  - `delta_idt_transfer = {_safe_float(best_clip.get('delta_idt_transfer', 0.0)):.4f}`",
        "- New `wikiarts5` best transfer-`LPIPS`:",
        f"  - `step = {best_lpips.get('step', '')}`",
        f"  - `transfer CLIP-S / LPIPS = {_safe_float(best_lpips.get('transfer_clip_style', 0.0)):.4f} / {_safe_float(best_lpips.get('transfer_lpips', 0.0)):.4f}`",
        f"  - `delta_idt_transfer = {_safe_float(best_lpips.get('delta_idt_transfer', 0.0)):.4f}`",
        "- Latest settled checkpoint:",
        f"  - `step = {latest.get('step', '')}`",
        f"  - `transfer CLIP-S / LPIPS = {_safe_float(latest.get('transfer_clip_style', 0.0)):.4f} / {_safe_float(latest.get('transfer_lpips', 0.0)):.4f}`",
        f"  - `delta_idt_transfer = {_safe_float(latest.get('delta_idt_transfer', 0.0)):.4f}`",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the WikiArts-5 baseline repro doc from live curve artifacts.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--wsl-distro", default="f")
    parser.add_argument("--page1-doc-path", type=Path, default=DEFAULT_PAGE1_DOC_PATH)
    parser.add_argument("--page1-script-path", type=Path, default=DEFAULT_PAGE1_SCRIPT_PATH)
    parser.add_argument("--page1-summary-json", type=Path, default=DEFAULT_PAGE1_SUMMARY_JSON)
    parser.add_argument("--page1-summary-csv", type=Path, default=DEFAULT_PAGE1_SUMMARY_CSV)
    parser.add_argument("--page1-curve-csv", type=Path, default=DEFAULT_PAGE1_CURVE_CSV)
    parser.add_argument("--page1-summary-png", type=Path, default=DEFAULT_PAGE1_SUMMARY_PNG)
    parser.add_argument("--page1-qual-png", type=Path, default=DEFAULT_PAGE1_QUAL_PNG)
    args = parser.parse_args()

    result_root = Path(str(args.result_root)).expanduser()
    doc_path = Path(str(args.doc_path)).expanduser()
    page1_doc_path = Path(str(args.page1_doc_path)).expanduser()
    page1_script_path = Path(str(args.page1_script_path)).expanduser()
    page1_summary_json = Path(str(args.page1_summary_json)).expanduser()
    page1_summary_csv = Path(str(args.page1_summary_csv)).expanduser()
    page1_curve_csv = Path(str(args.page1_curve_csv)).expanduser()
    page1_summary_png = Path(str(args.page1_summary_png)).expanduser()
    page1_qual_png = Path(str(args.page1_qual_png)).expanduser()
    curve_csv = result_root / "curve_metrics.csv"
    convergence_json = result_root / "curve_convergence.json"
    curve_rows = _read_rows(curve_csv) if curve_csv.exists() else []
    convergence = _read_json(convergence_json) if convergence_json.exists() else None
    if curve_rows and (convergence is None or int(convergence.get("row_count", -1)) != len(curve_rows)):
        style_key = "transfer_clip_style" if "transfer_clip_style" in curve_rows[0] else "clip_style"
        lpips_key = "transfer_lpips" if "transfer_lpips" in curve_rows[0] else "content_lpips"
        convergence = _compute_convergence(curve_rows, style_key=style_key, lpips_key=lpips_key)
        convergence["curve_csv"] = str(curve_csv)
        convergence_json.write_text(json.dumps(convergence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gpu = _run_local_gpu_query()
    processes = _query_wsl_processes(distro=str(args.wsl_distro), result_root=result_root)

    payload = {
        "result_root": str(result_root),
        "row_count": len(curve_rows),
        "latest_step": None if not curve_rows else _i(_latest_row(curve_rows), "step"),
        "best_clip_step": None if not curve_rows else _i(_best_row(curve_rows, prefer="clip"), "step"),
        "best_lpips_step": None if not curve_rows else _i(_best_row(curve_rows, prefer="lpips"), "step"),
        "convergence": convergence,
        "local_gpu": gpu,
        "active_processes": processes,
    }
    (result_root / "baseline_live_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _upsert_auto_block(
        doc_path,
        _render_auto_status(
            result_root=result_root,
            curve_rows=curve_rows,
            convergence=convergence,
            gpu=gpu,
            processes=processes,
        ),
    )
    page1_result = _run_python_script(page1_script_path) if page1_script_path.exists() else None
    if page1_result is not None:
        if page1_result.stdout.strip():
            print(page1_result.stdout.rstrip())
        if page1_result.stderr.strip():
            print(page1_result.stderr.rstrip())
        if page1_result.returncode != 0:
            print(f"page1_refresh_failed returncode={page1_result.returncode} script={page1_script_path}")
    if page1_summary_json.exists():
        page1_summary = _read_json(page1_summary_json)
        _upsert_marked_block(
            page1_doc_path,
            _render_page1_auto_status(
                summary=page1_summary,
                summary_json_path=page1_summary_json,
                summary_csv_path=page1_summary_csv,
                curve_csv_path=page1_curve_csv,
                summary_png_path=page1_summary_png,
                qual_png_path=page1_qual_png,
            ),
            start_marker=PAGE1_AUTO_START,
            end_marker=PAGE1_AUTO_END,
        )
    print(doc_path)
    print(result_root / "baseline_live_status.json")
    if page1_doc_path.exists():
        print(page1_doc_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
