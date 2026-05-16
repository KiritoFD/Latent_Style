from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _default_test_dir(root: Path) -> Path:
    return (root.parent / "style_data" / "overfit50").resolve()


def _find_epoch_dirs(eval_root: Path) -> list[Path]:
    out = []
    for child in sorted(eval_root.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir() and (child / "summary.json").is_file() and (child / "metrics.csv").is_file():
            out.append(child)
    return out


def _summary_metrics(summary_path: Path) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis", {}) or {}
    all_pairs = analysis.get("all_pairs_overview", {}) or {}
    transfer = analysis.get("style_transfer_ability", {}) or {}
    photo = analysis.get("photo_to_art_performance", {}) or {}
    identity = analysis.get("identity_reconstruction", {}) or {}
    return {
        "checkpoint": str(payload.get("checkpoint", "")),
        "clip_style": all_pairs.get("clip_style"),
        "clip_content": all_pairs.get("clip_content"),
        "content_lpips": all_pairs.get("content_lpips"),
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
        "cmmd_all": all_pairs.get("cmmd"),
        "dino_structure_all": all_pairs.get("dino_structure"),
        "gram_micro_all": all_pairs.get("gram_micro"),
        "gram_macro_all": all_pairs.get("gram_macro"),
        "cmmd_transfer": transfer.get("cmmd"),
        "dino_structure_transfer": transfer.get("dino_structure"),
        "gram_micro_transfer": transfer.get("gram_micro"),
        "gram_macro_transfer": transfer.get("gram_macro"),
        "cmmd_photo_to_art": photo.get("cmmd"),
        "dino_structure_photo_to_art": photo.get("dino_structure"),
        "gram_micro_photo_to_art": photo.get("gram_micro"),
        "gram_macro_photo_to_art": photo.get("gram_macro"),
        "cmmd_identity": identity.get("cmmd"),
        "dino_structure_identity": identity.get("dino_structure"),
        "gram_micro_identity": identity.get("gram_micro"),
        "gram_macro_identity": identity.get("gram_macro"),
    }


def _write_batch_summary(eval_root: Path, rows: list[dict[str, object]]) -> None:
    json_path = eval_root / "batch_summary.json"
    csv_path = eval_root / "batch_summary.csv"
    viewer_csv_path = eval_root / "batch_summary_viewer.csv"
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
        "cmmd_all",
        "dino_structure_all",
        "gram_micro_all",
        "gram_macro_all",
        "cmmd_transfer",
        "dino_structure_transfer",
        "gram_micro_transfer",
        "gram_macro_transfer",
        "cmmd_photo_to_art",
        "dino_structure_photo_to_art",
        "gram_micro_photo_to_art",
        "gram_macro_photo_to_art",
        "cmmd_identity",
        "dino_structure_identity",
        "gram_micro_identity",
        "gram_macro_identity",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    viewer_fields = [
        "experiment_id",
        "epoch",
        "clip_style",
        "clip_content",
        "content_lpips",
        "cmmd_all",
        "dino_structure_all",
        "gram_micro_all",
        "gram_macro_all",
    ]
    with viewer_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=viewer_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in viewer_fields})


def main() -> int:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from utils.modern_metrics import ModernMetricConfig, append_modern_metrics_to_summary

    parser = argparse.ArgumentParser(description="Append modern post-hoc metrics to an existing full_eval directory.")
    parser.add_argument("eval_root", help="Path to full_eval directory containing epoch_xxxx subdirs.")
    parser.add_argument("--test_dir", default=str(_default_test_dir(root)), help="Dataset root used to resolve source/style images.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32", help="CLIP model/path used for CMMD.")
    parser.add_argument("--dino_model_name", default="facebook/dinov2-small", help="DINOv2 model name/path.")
    parser.add_argument("--cmmd_sigma", type=float, default=10.0, help="RBF sigma for CMMD.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction.")
    args = parser.parse_args()

    eval_root = Path(args.eval_root).resolve()
    epoch_dirs = _find_epoch_dirs(eval_root)
    if not epoch_dirs:
        raise SystemExit(f"No epoch dirs with summary.json + metrics.csv found under: {eval_root}")

    cfg = ModernMetricConfig(
        test_dir=Path(args.test_dir).resolve(),
        device=args.device,
        clip_model_name=args.clip_model_name,
        dino_model_name=args.dino_model_name,
        cmmd_sigma=float(args.cmmd_sigma),
        batch_size=int(args.batch_size),
    )

    rows: list[dict[str, object]] = []
    experiment_id = eval_root.parent.name if eval_root.name.lower() == "full_eval" else eval_root.name
    print(f"Append modern metrics | eval root: {eval_root}")
    print(f"Append modern metrics | epochs: {len(epoch_dirs)}")
    for epoch_dir in epoch_dirs:
        print(f"  -> {epoch_dir.name}")
        payload = append_modern_metrics_to_summary(epoch_dir, cfg)
        row: dict[str, object] = {
            "experiment_id": experiment_id,
            "epoch": epoch_dir.name,
            "checkpoint_path": str(Path(payload.get("checkpoint", ""))),
            "output_dir": str(epoch_dir),
            "status": "updated_modern_metrics",
            "returncode": 0,
            "summary_exists": True,
        }
        row.update(_summary_metrics(epoch_dir / "summary.json"))
        rows.append(row)

    _write_batch_summary(eval_root, rows)
    print(f"Append modern metrics finished | summary: {eval_root / 'batch_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
