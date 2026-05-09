from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _has_flag(argv: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


def _flag_value(argv: list[str], name: str) -> str | None:
    for idx, arg in enumerate(argv):
        if arg == name and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return None


def _strip_flag(argv: list[str], name: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == name:
            if idx + 1 < len(argv) and not argv[idx + 1].startswith("-"):
                skip_next = True
            continue
        if arg.startswith(f"{name}="):
            continue
        out.append(arg)
    return out


def _summary_already_exists(out_dir: Path) -> bool:
    return (out_dir / "summary.json").is_file()


def _inject_defaults(root: Path, argv: list[str]) -> list[str]:
    workspace = root.parent
    local_clip_dir = workspace / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32"
    default_test_dir = workspace / "style_data" / "overfit50"
    default_cache_dir = workspace / "Cycle-NCE" / "eval_cache"
    default_clip_hf_cache_dir = default_cache_dir / "hf"

    has_clip_model_name = _has_flag(argv, "--clip_model_name")
    has_clip_backend = _has_flag(argv, "--clip_backend")
    has_test_dir = _has_flag(argv, "--test_dir")
    has_cache_dir = _has_flag(argv, "--cache_dir")
    has_clip_hf_cache_dir = _has_flag(argv, "--clip_hf_cache_dir")

    out = list(argv)
    if not has_clip_backend:
        out.extend(["--clip_backend", "hf"])
    if not has_clip_model_name and local_clip_dir.exists():
        out.extend(["--clip_model_name", str(local_clip_dir)])
    if not has_test_dir and default_test_dir.exists():
        out.extend(["--test_dir", str(default_test_dir)])
    if not has_cache_dir and default_cache_dir.exists():
        out.extend(["--cache_dir", str(default_cache_dir)])
    if not has_clip_hf_cache_dir and default_clip_hf_cache_dir.exists():
        out.extend(["--clip_hf_cache_dir", str(default_clip_hf_cache_dir)])
    return out


def _single_eval(root: Path, argv: list[str]) -> int:
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    sys.argv = [sys.argv[0], *argv]
    import runpy

    runpy.run_module("utils.run_evaluation", run_name="__main__")
    return 0


def _find_ckpts(ckpt_dir: Path) -> list[Path]:
    return sorted(
        [p for p in ckpt_dir.glob("epoch_*.pt") if p.is_file()],
        key=lambda p: p.name.lower(),
    )


def _find_experiment_dirs(parent_dir: Path) -> list[Path]:
    out: list[Path] = []
    for child in sorted(parent_dir.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if _find_ckpts(child):
            out.append(child)
    return out


def _summary_metrics(summary_path: Path) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis", {})
    all_pairs = analysis.get("all_pairs_overview", {}) or {}
    transfer = analysis.get("style_transfer_ability", {}) or {}
    photo = analysis.get("photo_to_art_performance", {}) or {}
    return {
        "checkpoint": str(payload.get("checkpoint", "")),
        "clip_style": transfer.get("clip_style"),
        "clip_content": transfer.get("clip_content"),
        "content_lpips": transfer.get("content_lpips"),
        "all_clip_style": all_pairs.get("clip_style"),
        "all_clip_content": all_pairs.get("clip_content"),
        "all_content_lpips": all_pairs.get("content_lpips"),
        "transfer_clip_style": transfer.get("clip_style"),
        "transfer_clip_content": transfer.get("clip_content"),
        "transfer_content_lpips": transfer.get("content_lpips"),
        "clip_style_all": all_pairs.get("clip_style"),
        "clip_content_all": all_pairs.get("clip_content"),
        "content_lpips_all": all_pairs.get("content_lpips"),
        "clip_style_transfer": transfer.get("clip_style"),
        "clip_content_transfer": transfer.get("clip_content"),
        "content_lpips_transfer": transfer.get("content_lpips"),
        "clip_style_photo_to_art": photo.get("clip_style"),
        "clip_content_photo_to_art": photo.get("clip_content"),
        "content_lpips_photo_to_art": photo.get("content_lpips"),
    }


def _write_batch_summary(out_root: Path, rows: list[dict[str, object]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "batch_summary.json"
    csv_path = out_root / "batch_summary.csv"
    viewer_csv_path = out_root / "batch_summary_viewer.csv"
    json_path.write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fieldnames = [
        "experiment_id",
        "epoch",
        "checkpoint_path",
        "output_dir",
        "status",
        "returncode",
        "summary_exists",
        "checkpoint",
        "clip_style",
        "clip_content",
        "content_lpips",
        "all_clip_style",
        "all_clip_content",
        "all_content_lpips",
        "transfer_clip_style",
        "transfer_clip_content",
        "transfer_content_lpips",
        "clip_style_all",
        "clip_content_all",
        "content_lpips_all",
        "clip_style_transfer",
        "clip_content_transfer",
        "content_lpips_transfer",
        "clip_style_photo_to_art",
        "clip_content_photo_to_art",
        "content_lpips_photo_to_art",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    viewer_fieldnames = [
        "experiment_id",
        "epoch",
        "clip_style",
        "clip_content",
        "content_lpips",
        "all_clip_style",
        "all_clip_content",
        "all_content_lpips",
        "transfer_clip_style",
        "transfer_clip_content",
        "transfer_content_lpips",
    ]
    with viewer_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=viewer_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in viewer_fieldnames})


def _collect_batch_rows(root: Path, ckpt_dir: Path, output_root: Path, passthrough: list[str]) -> tuple[list[dict[str, object]], int]:
    ckpts = _find_ckpts(ckpt_dir)
    rows: list[dict[str, object]] = []
    fail_count = 0
    experiment_id = ckpt_dir.name
    force_rerun = _has_flag(passthrough, "--force")
    passthrough = _strip_flag(passthrough, "--force")
    for ckpt in ckpts:
        epoch_name = ckpt.stem
        out_dir = output_root / epoch_name
        summary_path = out_dir / "summary.json"
        if summary_path.exists() and not force_rerun:
            print(f"\n[{epoch_name}] skip existing -> {out_dir}")
            row: dict[str, object] = {
                "experiment_id": experiment_id,
                "epoch": epoch_name,
                "checkpoint_path": str(ckpt),
                "output_dir": str(out_dir),
                "status": "skipped_existing",
                "returncode": 0,
                "summary_exists": True,
            }
            row.update(_summary_metrics(summary_path))
            rows.append(row)
            continue

        cmd = [sys.executable, str(root / "run_evaluation.py"), str(ckpt), "--output", str(out_dir), *passthrough]
        print(f"\n[{epoch_name}] eval -> {out_dir}")
        result = subprocess.run(cmd, cwd=root)
        row: dict[str, object] = {
            "experiment_id": experiment_id,
            "epoch": epoch_name,
            "checkpoint_path": str(ckpt),
            "output_dir": str(out_dir),
            "status": "ok" if result.returncode == 0 else "fail",
            "returncode": result.returncode,
            "summary_exists": summary_path.exists(),
        }
        if summary_path.exists():
            row.update(_summary_metrics(summary_path))
        if result.returncode != 0:
            fail_count += 1
        rows.append(row)
    return rows, fail_count


def _batch_eval(root: Path, ckpt_dir: Path, argv: list[str]) -> int:
    ckpts = _find_ckpts(ckpt_dir)
    if not ckpts:
        print(f"No checkpoint files found under: {ckpt_dir}")
        return 1

    output_flag = _flag_value(argv, "--output")
    output_root = Path(output_flag).resolve() if output_flag else (ckpt_dir / "full_eval")
    passthrough = _strip_flag(argv, "--output")

    print(f"Batch eval | ckpt dir: {ckpt_dir}")
    print(f"Batch eval | output root: {output_root}")
    print(f"Batch eval | checkpoints: {len(ckpts)}")

    rows, fail_count = _collect_batch_rows(root, ckpt_dir, output_root, passthrough)

    _write_batch_summary(output_root, rows)
    print(f"\nBatch eval finished | failures: {fail_count} | summary: {output_root / 'batch_summary.csv'}")
    return 1 if fail_count > 0 else 0


def _multi_experiment_eval(root: Path, parent_dir: Path, argv: list[str]) -> int:
    exp_dirs = _find_experiment_dirs(parent_dir)
    if not exp_dirs:
        print(f"No experiment directories with epoch_*.pt found under: {parent_dir}")
        return 1

    output_flag = _flag_value(argv, "--output")
    output_root = Path(output_flag).resolve() if output_flag else (parent_dir / "full_eval")
    passthrough = _strip_flag(argv, "--output")

    print(f"Multi-experiment eval | parent dir: {parent_dir}")
    print(f"Multi-experiment eval | output root: {output_root}")
    print(f"Multi-experiment eval | experiments: {len(exp_dirs)}")

    all_rows: list[dict[str, object]] = []
    fail_count = 0
    for exp_dir in exp_dirs:
        exp_output_root = output_root / exp_dir.name
        print(f"\n== Experiment: {exp_dir.name} ==")
        rows, exp_fail = _collect_batch_rows(root, exp_dir, exp_output_root, passthrough)
        _write_batch_summary(exp_output_root, rows)
        all_rows.extend(rows)
        fail_count += exp_fail

    _write_batch_summary(output_root, all_rows)
    print(f"\nMulti-experiment eval finished | failures: {fail_count} | summary: {output_root / 'batch_summary.csv'}")
    return 1 if fail_count > 0 else 0


def main() -> None:
    root = Path(__file__).resolve().parent
    argv = list(sys.argv[1:])
    positional = [arg for arg in argv if arg and not arg.startswith("-")]
    has_checkpoint_flag = _has_flag(argv, "--checkpoint")
    normalized = _inject_defaults(root, argv)

    if positional and not has_checkpoint_flag:
        first = Path(positional[0]).resolve()
        if first.is_dir():
            if _find_ckpts(first):
                raise SystemExit(_batch_eval(root, first, normalized))
            if _find_experiment_dirs(first):
                raise SystemExit(_multi_experiment_eval(root, first, normalized))
            raise SystemExit(_batch_eval(root, first, normalized))
        if first.suffix.lower() == ".pt":
            remainder = normalized[1:]
            normalized = ["--checkpoint", str(first), *remainder]
            if not _has_flag(normalized, "--output"):
                default_output = str(first.parent / "full_eval" / first.stem)
                normalized.extend(["--output", default_output])

    if has_checkpoint_flag:
        ckpt_value = _flag_value(normalized, "--checkpoint")
        if ckpt_value:
            ckpt_path = Path(ckpt_value).resolve()
            if ckpt_path.is_dir():
                if _find_ckpts(ckpt_path):
                    raise SystemExit(_batch_eval(root, ckpt_path, normalized))
                if _find_experiment_dirs(ckpt_path):
                    raise SystemExit(_multi_experiment_eval(root, ckpt_path, normalized))
                raise SystemExit(_batch_eval(root, ckpt_path, normalized))
            if not _has_flag(normalized, "--output"):
                normalized.extend(["--output", str(ckpt_path.parent / "full_eval" / ckpt_path.stem)])

    raise SystemExit(_single_eval(root, normalized))


if __name__ == "__main__":
    main()
