from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

EPOCH_DIR_RE = re.compile(r"^epoch_(\d+)(?:_(tokenized))?$")


def _to_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def _discover_eval_dirs(roots: list[Path]) -> list[Path]:
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            print(f"[WARN] root not found, skip: {root}")
            continue
        for images_dir in root.rglob("images"):
            if not images_dir.is_dir():
                continue
            epoch_dir = images_dir.parent
            full_eval_dir = epoch_dir.parent
            if full_eval_dir.name != "full_eval":
                continue
            if EPOCH_DIR_RE.fullmatch(epoch_dir.name) is None:
                continue
            found[str(epoch_dir.resolve())] = epoch_dir.resolve()
    return sorted(found.values(), key=lambda p: str(p))


def _parse_eval_meta(eval_dir: Path) -> dict[str, Any]:
    m = EPOCH_DIR_RE.fullmatch(eval_dir.name)
    if m is None:
        raise ValueError(f"Invalid epoch dir: {eval_dir}")
    epoch = int(m.group(1))
    variant = "tokenized" if m.group(2) else "original"

    exp_name_base = eval_dir.parent.parent.name
    exp_name = f"{exp_name_base}_tokenized" if variant == "tokenized" else exp_name_base
    run_id = f"{exp_name}/{eval_dir.name}"

    return {
        "run_id": run_id,
        "experiment_name": exp_name,
        "experiment_name_base": exp_name_base,
        "variant": variant,
        "epoch": epoch,
        "epoch_dir": eval_dir.name,
        "eval_dir": str(eval_dir),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _checkpoint_from_existing_summary(eval_dir: Path) -> Path | None:
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        data = _read_json(summary_path)
    except Exception:
        return None

    ckpt = str(data.get("checkpoint", "")).strip()
    if not ckpt:
        return None
    p = Path(ckpt)
    if p.exists():
        return p.resolve()
    return None


def _matrix_clip_content(summary: dict[str, Any], *, photo_to_art_only: bool) -> float | None:
    matrix = summary.get("matrix_breakdown", {})
    if not isinstance(matrix, dict):
        return None

    vals: list[float] = []
    for src_style, tgt_map in matrix.items():
        if not isinstance(tgt_map, dict):
            continue
        for tgt_style, item in tgt_map.items():
            if not isinstance(item, dict):
                continue
            if photo_to_art_only:
                if str(src_style).lower() != "photo" or str(tgt_style).lower() == "photo":
                    continue
            v = _to_float(item.get("clip_content"))
            if v is not None:
                vals.append(v)
    return _mean(vals)


def _extract_row(eval_dir: Path, summary: dict[str, Any], status: str, error: str = "") -> dict[str, Any]:
    meta = _parse_eval_meta(eval_dir)
    analysis = summary.get("analysis", {}) if isinstance(summary, dict) else {}
    sta = analysis.get("style_transfer_ability", {}) if isinstance(analysis, dict) else {}
    p2a = analysis.get("photo_to_art_performance", {}) if isinstance(analysis, dict) else {}

    sta_clip_content = _to_float(sta.get("clip_content"))
    if sta_clip_content is None:
        sta_clip_content = _matrix_clip_content(summary, photo_to_art_only=False)

    p2a_clip_content = _to_float(p2a.get("clip_content"))
    if p2a_clip_content is None:
        p2a_clip_content = _matrix_clip_content(summary, photo_to_art_only=True)

    return {
        **meta,
        "status": status,
        "error": error,
        "summary_path": str((eval_dir / "summary.json").resolve()),
        "timestamp": summary.get("timestamp", "") if isinstance(summary, dict) else "",
        "checkpoint": summary.get("checkpoint", "") if isinstance(summary, dict) else "",
        "sta_clip_dir": sta.get("clip_dir") if isinstance(sta, dict) else "",
        "sta_clip_style": sta.get("clip_style") if isinstance(sta, dict) else "",
        "sta_content_lpips": sta.get("content_lpips") if isinstance(sta, dict) else "",
        "sta_clip_content": sta_clip_content,
        "sta_fid_baseline": sta.get("fid_baseline") if isinstance(sta, dict) else "",
        "sta_fid": sta.get("fid") if isinstance(sta, dict) else "",
        "sta_delta_fid": sta.get("delta_fid") if isinstance(sta, dict) else "",
        "sta_delta_fid_ratio": sta.get("delta_fid_ratio") if isinstance(sta, dict) else "",
        "sta_art_fid": sta.get("art_fid") if isinstance(sta, dict) else "",
        "sta_kid_baseline": sta.get("kid_baseline") if isinstance(sta, dict) else "",
        "sta_kid": sta.get("kid") if isinstance(sta, dict) else "",
        "sta_delta_kid": sta.get("delta_kid") if isinstance(sta, dict) else "",
        "sta_delta_kid_ratio": sta.get("delta_kid_ratio") if isinstance(sta, dict) else "",
        "sta_classifier_acc": sta.get("classifier_acc") if isinstance(sta, dict) else "",
        "p2a_valid": p2a.get("valid") if isinstance(p2a, dict) else "",
        "p2a_clip_dir": p2a.get("clip_dir") if isinstance(p2a, dict) else "",
        "p2a_clip_style": p2a.get("clip_style") if isinstance(p2a, dict) else "",
        "p2a_clip_content": p2a_clip_content,
        "p2a_fid_baseline": p2a.get("fid_baseline") if isinstance(p2a, dict) else "",
        "p2a_fid": p2a.get("fid") if isinstance(p2a, dict) else "",
        "p2a_delta_fid": p2a.get("delta_fid") if isinstance(p2a, dict) else "",
        "p2a_delta_fid_ratio": p2a.get("delta_fid_ratio") if isinstance(p2a, dict) else "",
        "p2a_art_fid": p2a.get("art_fid") if isinstance(p2a, dict) else "",
        "p2a_kid_baseline": p2a.get("kid_baseline") if isinstance(p2a, dict) else "",
        "p2a_kid": p2a.get("kid") if isinstance(p2a, dict) else "",
        "p2a_delta_kid": p2a.get("delta_kid") if isinstance(p2a, dict) else "",
        "p2a_delta_kid_ratio": p2a.get("delta_kid_ratio") if isinstance(p2a, dict) else "",
        "p2a_classifier_acc": p2a.get("classifier_acc") if isinstance(p2a, dict) else "",
    }


def _extract_or_placeholder(eval_dir: Path, status: str, error: str = "") -> dict[str, Any]:
    summary_path = eval_dir / "summary.json"
    if summary_path.exists():
        try:
            return _extract_row(eval_dir, _read_json(summary_path), status=status, error=error)
        except Exception as e:
            return _extract_row(eval_dir, {}, status="failed", error=f"summary_parse_error: {e}")
    return _extract_row(eval_dir, {}, status=status, error=error)


def _load_summary_if_exists(eval_dir: Path) -> dict[str, Any] | None:
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return _read_json(summary_path)
    except Exception:
        return None


def _has_metric_value(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"", "none", "null", "nan"}:
            return False
    return True


def _missing_required_metrics(summary: dict[str, Any] | list[Any] | None) -> list[str]:
    if not isinstance(summary, dict):
        return ["summary_missing"]
    analysis = summary.get("analysis", {})
    if not isinstance(analysis, dict):
        return ["analysis_missing"]

    missing: list[str] = []
    sta = analysis.get("style_transfer_ability", {})
    if not isinstance(sta, dict):
        return ["analysis.style_transfer_ability_missing"]

    # Core transfer metrics we care about in this pipeline.
    required_sta = ["clip_dir", "clip_style", "content_lpips", "clip_content"]
    for k in required_sta:
        if not _has_metric_value(sta.get(k)):
            missing.append(f"sta.{k}")

    p2a = analysis.get("photo_to_art_performance", {})
    if isinstance(p2a, dict):
        p2a_valid = bool(p2a.get("valid", False))
        # Only require p2a metrics when p2a section is marked valid.
        if p2a_valid:
            required_p2a = ["clip_dir", "clip_style", "clip_content"]
            for k in required_p2a:
                if not _has_metric_value(p2a.get(k)):
                    missing.append(f"p2a.{k}")

    return missing


def _run_eval(src_dir: Path, eval_dir: Path, checkpoint: Path | None, test_dir: str, dry_run: bool) -> tuple[bool, str]:
    cmd = [
        "uv",
        "run",
        "python",
        "utils/run_evaluation.py",
        "--output",
        str(eval_dir),
        "--reuse_generated",
        "--force_regen",
    ]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    if test_dir.strip():
        cmd += ["--test_dir", test_dir.strip()]

    print(f"[RUN] eval  {eval_dir}")
    print("      " + " ".join(cmd))
    if dry_run:
        return True, ""

    try:
        subprocess.run(cmd, check=True, cwd=str(src_dir))
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, f"exit_code={e.returncode}"


def _run_distill(
    src_dir: Path,
    original_eval_dir: Path,
    checkpoint: Path,
    distill_epochs: int,
    dry_run: bool,
) -> tuple[bool, str, Path]:
    exp_dir = original_eval_dir.parent.parent
    epoch_dir_name = original_eval_dir.name
    tokenized_eval_dir = original_eval_dir.parent / f"{epoch_dir_name}_tokenized"
    tokenized_eval_dir.mkdir(parents=True, exist_ok=True)

    distill_out = exp_dir / "tokenizer_distill" / epoch_dir_name
    distill_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "python",
        "prob.py",
        "--checkpoint",
        str(checkpoint),
        "--output_dir",
        str(distill_out),
        "--epochs",
        str(int(distill_epochs)),
        "--run_full_eval",
        "--full_eval_output",
        str(tokenized_eval_dir),
    ]

    print(f"[RUN] distill {original_eval_dir.name} -> {tokenized_eval_dir.name}")
    print("      " + " ".join(cmd))
    if dry_run:
        return True, "", tokenized_eval_dir

    try:
        subprocess.run(cmd, check=True, cwd=str(src_dir))
        return True, "", tokenized_eval_dir
    except subprocess.CalledProcessError as e:
        return False, f"exit_code={e.returncode}", tokenized_eval_dir


def _write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "experiment_name",
        "experiment_name_base",
        "variant",
        "epoch",
        "epoch_dir",
        "eval_dir",
        "summary_path",
        "timestamp",
        "checkpoint",
        "status",
        "error",
        "sta_clip_dir",
        "sta_clip_style",
        "sta_content_lpips",
        "sta_clip_content",
        "sta_fid_baseline",
        "sta_fid",
        "sta_delta_fid",
        "sta_delta_fid_ratio",
        "sta_art_fid",
        "sta_kid_baseline",
        "sta_kid",
        "sta_delta_kid",
        "sta_delta_kid_ratio",
        "sta_classifier_acc",
        "p2a_valid",
        "p2a_clip_dir",
        "p2a_clip_style",
        "p2a_clip_content",
        "p2a_fid_baseline",
        "p2a_fid",
        "p2a_delta_fid",
        "p2a_delta_fid_ratio",
        "p2a_art_fid",
        "p2a_kid_baseline",
        "p2a_kid",
        "p2a_delta_kid",
        "p2a_delta_kid_ratio",
        "p2a_classifier_acc",
    ]

    rows_sorted = sorted(rows, key=lambda r: (str(r.get("experiment_name_base", "")), int(r.get("epoch", 0) or 0), str(r.get("variant", ""))))

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find full_eval/epoch_*(_tokenized)/images under roots, "
            "rerun full_eval with --reuse_generated, optionally distill all non-tokenized runs to 3000 epochs, "
            "then export summary CSV."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[r"G:\GitHub\Latent_Style\Cycle-NCE", r"Y:\experiments"],
        help="Root directories to scan recursively",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default=r"G:\GitHub\Latent_Style\Related_Works\runs_eval_summary_reuse.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="",
        help="Optional override passed to run_evaluation.py --test_dir",
    )
    parser.add_argument("--distill_epochs", type=int, default=3000, help="prob.py distill epochs for non-tokenized runs")
    parser.add_argument("--skip_distill", action="store_true", help="Skip distill on non-tokenized runs")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N discovered eval dirs")
    parser.add_argument("--dry_run", action="store_true", help="Print commands, do not execute")
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    roots = [Path(x).expanduser().resolve() for x in args.roots]
    csv_out = Path(args.csv_out).expanduser().resolve()

    eval_dirs = _discover_eval_dirs(roots)
    if args.limit > 0:
        eval_dirs = eval_dirs[: int(args.limit)]

    print(f"[INFO] discovered eval dirs: {len(eval_dirs)}")
    if not eval_dirs:
        _write_csv([], csv_out)
        print(f"[OK] empty result written: {csv_out}")
        return

    rows_by_eval: dict[str, dict[str, Any]] = {}

    for idx, eval_dir in enumerate(eval_dirs, start=1):
        try:
            meta = _parse_eval_meta(eval_dir)
            print(f"\n[{idx}/{len(eval_dirs)}] {eval_dir}  ({meta['variant']})")

            existing_summary = _load_summary_if_exists(eval_dir)
            missing_eval_metrics = _missing_required_metrics(existing_summary)
            if not missing_eval_metrics:
                print("  skip eval: all required metrics already exist")
                rows_by_eval[str(eval_dir.resolve())] = _extract_or_placeholder(
                    eval_dir,
                    status="skip_metric_exists",
                    error="",
                )
                checkpoint = _checkpoint_from_existing_summary(eval_dir)
            else:
                print(f"  rerun eval: missing metrics -> {', '.join(missing_eval_metrics)}")
                checkpoint = _checkpoint_from_existing_summary(eval_dir)
                if checkpoint is not None:
                    print(f"  checkpoint: {checkpoint}")
                else:
                    print("  checkpoint: (none; reuse-only mode)")
                ok, err = _run_eval(src_dir, eval_dir, checkpoint, str(args.test_dir), bool(args.dry_run))
                status = "ok" if ok else "failed"
                if args.dry_run and ok:
                    status = "ok(dry_run)"
                rows_by_eval[str(eval_dir.resolve())] = _extract_or_placeholder(eval_dir, status=status, error=err)

            if meta["variant"] != "original" or args.skip_distill:
                continue

            if "decoder" not in str(meta.get("experiment_name_base", "")).lower():
                print("  skip distill: experiment name does not contain 'decoder'")
                continue

            tokenized_eval_dir = eval_dir.parent / f"{eval_dir.name}_tokenized"
            tokenized_existing = _load_summary_if_exists(tokenized_eval_dir)
            missing_tok_metrics = _missing_required_metrics(tokenized_existing)
            if not missing_tok_metrics:
                print("  skip distill: tokenized required metrics already exist")
                rows_by_eval[str(tokenized_eval_dir.resolve())] = _extract_or_placeholder(
                    tokenized_eval_dir,
                    status="skip_metric_exists",
                    error="",
                )
                continue

            if checkpoint is None:
                rows_by_eval[str(tokenized_eval_dir.resolve())] = _extract_or_placeholder(
                    tokenized_eval_dir,
                    status="skip_incompatible_or_missing_checkpoint",
                    error="distill_skip_no_checkpoint",
                )
                continue

            d_ok, d_err, tokenized_eval_dir = _run_distill(
                src_dir=src_dir,
                original_eval_dir=eval_dir,
                checkpoint=checkpoint,
                distill_epochs=int(args.distill_epochs),
                dry_run=bool(args.dry_run),
            )
            d_status = "ok" if d_ok else "skip_incompatible_or_failed"
            if args.dry_run and d_ok:
                d_status = "ok(dry_run)"
            err_msg = d_err if d_err else ""
            rows_by_eval[str(tokenized_eval_dir.resolve())] = _extract_or_placeholder(
                tokenized_eval_dir,
                status=d_status,
                error=err_msg,
            )
        except Exception as e:
            rows_by_eval[str(eval_dir.resolve())] = _extract_or_placeholder(
                eval_dir,
                status="skip_error",
                error=f"{type(e).__name__}: {e}",
            )
            continue

    rows = list(rows_by_eval.values())
    _write_csv(rows, csv_out)
    ok_count = sum(1 for r in rows if str(r.get("status", "")).startswith("ok"))
    print(f"\n[OK] csv written: {csv_out}")
    print(f"[OK] success={ok_count}, failed={len(rows) - ok_count}, total={len(rows)}")


if __name__ == "__main__":
    main()
