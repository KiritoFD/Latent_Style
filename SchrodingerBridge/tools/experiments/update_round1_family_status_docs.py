from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_config
from round1_registry import COMMON_PARENT_CONFIG
from round1_paths import infer_round1_family_id, round1_family_doc_dir, round1_fast_local_root, round1_localreview_root


AUTO_START = "<!-- ROUND1_AUTO_STATUS:START -->"
AUTO_END = "<!-- ROUND1_AUTO_STATUS:END -->"
DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"
DEFAULT_MASTER = SB_ROOT / "docs" / "experiments" / "2026-06-10-round1-full-sweep-master.md"
DEFAULT_LOCAL_GPU_LOCK = SB_ROOT / "aaai2027" / ".local_gpu_eval.lock"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, str]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
            new_text = text[:start] + block + text[end + len(AUTO_END) :]
        else:
            suffix = "" if text.endswith("\n") else "\n"
            new_text = text + suffix + "\n" + block
    else:
        title = f"# {path.stem.replace('_', ' ').title()}\n\n"
        new_text = title + block
    path.write_text(new_text, encoding="utf-8")


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
    curve_rows: list[dict[str, str]],
    convergence: dict | None,
) -> str:
    if not curve_rows:
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
        f"- Curve CSV: {_md_link('clip_lpips_curve.csv', fast_root / 'full_eval_fast_local' / 'clip_lpips_curve.csv')}",
    ]
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
    if pending:
        lines.extend(["- Pending pulled checkpoints:", *[f"  - `{name}`" for name in pending[-3:]]])
    if convergence:
        lines.extend(
            [
                "- Convergence snapshot:",
                f"  - `row_count = {convergence.get('row_count')}`",
                f"  - `best_epoch = {convergence.get('best_epoch')}`",
                f"  - `since_best = {convergence.get('since_best')}`",
                f"  - `best_in_newest_2 = {convergence.get('best_in_newest_2')}`",
                f"  - `tail_flat = {convergence.get('tail_flat')}`",
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
) -> str:
    run_name = str(row.get("run_name", "")).strip()
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
    if pending:
        lines.extend(["- Pending local fast eval:", *[f"  - `{name}`" for name in pending[-3:]]])
    return "\n".join(lines)


def _render_master_auto(
    *,
    running_rows: list[dict[str, str]],
    family_row: dict[str, str],
    curve_rows: list[dict[str, str]],
    convergence: dict | None,
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
    lines.extend(
        [
            f"- Active family: `{family_row.get('family_id', '')}`",
            f"- Decision status: `{family_row.get('decision_status', '')}`",
            f"- Batch / epochs / patience: `{family_row.get('batch_size', '')} / {family_row.get('num_epochs', '')} / {family_row.get('patience', '')}`",
        ]
    )
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
            f"- Convergence: `row_count={convergence.get('row_count')}, since_best={convergence.get('since_best')}, tail_flat={convergence.get('tail_flat')}, converged={convergence.get('converged')}`"
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
        "local_fast_root",
        "local_review_root",
        "best_ckpt",
        "best_transfer_lpips_ckpt",
        "best_allpairs_clip_style_ckpt",
        "latest_ckpt",
        "fast_converged",
        "convergence_reason",
        "decision_status",
    ]
    extras: list[str] = []
    seen = set(preferred)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                extras.append(key)
                seen.add(key)
    return preferred + extras


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
    parser.add_argument("--fast-eval-subdir", default="full_eval_fast_local")
    parser.add_argument("--review-eval-subdir", default="full_eval_fresh_localreview")
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
    fast_eval_root = fast_root / str(args.fast_eval_subdir).strip()

    curve_csv = fast_eval_root / "clip_lpips_curve.csv"
    convergence_json = fast_eval_root / "round1_convergence.json"
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

    family_row["parent_config"] = str(COMMON_PARENT_CONFIG)
    family_row["tokenizer_family"] = str(((cfg.get("model") or {}).get("tokenizer_family", "legacy_factorized")))
    family_row["backbone_attention_family"] = str(((cfg.get("model") or {}).get("backbone_attention_family", "legacy_semantic_crossattn")))
    family_row["solver_family"] = str(((cfg.get("model") or {}).get("solver_family", "euler_legacy")))
    family_row["semantic_supervision_family"] = str(((cfg.get("bridge") or {}).get("semantic_supervision_family", "legacy_terminal_swd")))
    family_row["local_fast_root"] = str(fast_root)
    family_row["local_review_root"] = str(localreview_root)
    family_row["best_ckpt"] = "" if convergence is None else str(convergence.get("best_epoch", ""))
    family_row["best_transfer_lpips_ckpt"] = "" if best_transfer_lpips is None else str(best_transfer_lpips.get("epoch", ""))
    family_row["best_allpairs_clip_style_ckpt"] = "" if best_full_style is None else str(best_full_style.get("epoch", ""))
    family_row["latest_ckpt"] = "" if latest is None else str(latest.get("epoch", ""))
    family_row["fast_converged"] = "" if convergence is None else str(convergence.get("converged", ""))
    if convergence is None:
        family_row["convergence_reason"] = ""
    else:
        family_row["convergence_reason"] = (
            f"best_in_newest_2={convergence.get('best_in_newest_2')}; "
            f"since_best={convergence.get('since_best')}; "
            f"tail_flat={convergence.get('tail_flat')}; "
            f"patience={convergence.get('patience')}"
        )
    _merge_family_row_into_manifest(manifest, family_id=family_id, updated_row=family_row)
    rows = _read_rows(manifest)

    _upsert_auto_block(
        family_doc_dir / "fast_curve_read.md",
        _render_fast_curve_auto(
            family_id=family_id,
            fast_root=fast_root,
            curve_rows=curve_rows,
            convergence=convergence,
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
        ),
    )
    running_rows = [row for row in rows if str(row.get("decision_status", "")).strip().lower() == "running"]
    master_row = running_rows[0] if running_rows else family_row
    _, master_fast_root, _ = _family_runtime_paths(master_row)
    master_curve_csv = master_fast_root / str(args.fast_eval_subdir).strip() / "clip_lpips_curve.csv"
    master_convergence_json = master_fast_root / str(args.fast_eval_subdir).strip() / "round1_convergence.json"
    master_curve_rows = _read_rows(master_curve_csv) if master_curve_csv.exists() else []
    master_convergence = _read_json(master_convergence_json) if master_convergence_json.exists() else None
    _upsert_auto_block(
        Path(args.master_note).resolve(),
        _render_master_auto(
            running_rows=running_rows,
            family_row=master_row,
            curve_rows=master_curve_rows,
            convergence=master_convergence,
        ),
    )
    print(family_doc_dir)
    print(Path(args.master_note).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
