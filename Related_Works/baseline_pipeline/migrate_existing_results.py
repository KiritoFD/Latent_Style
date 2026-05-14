from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
STYLE_NAMES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]


DEFAULT_SOURCES = {
    "ours_pareto_probe_4_epoch_0001": Path(
        "SchrodingerBridge/exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15/full_eval/epoch_0001/images"
    ),
    "cut": Path("Related_Works/runs/cut_5x5/infer_5x5/images"),
    "samst": Path("Related_Works/external/SaMST/full_eval/repro_5style_train2/epoch_0100_overfit50/images"),
    "sdturbo": Path("Related_Works/runs/sdturbo_5x5/images"),
    "sdedit_str_0p10": Path("Related_Works/runs/sdedit_multi/str_0.10/images"),
    "sdedit_str_0p20": Path("Related_Works/runs/sdedit_multi/str_0.20/images"),
    "sdedit_str_0p35": Path("Related_Works/runs/sdedit_multi/str_0.35/images"),
    "sdedit_str_0p40": Path("Related_Works/runs/sdedit_multi/str_0.40/images"),
}


def _reference_names(reference_images_dir: Path) -> set[str]:
    names = {p.name for p in reference_images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    if not names:
        raise RuntimeError(f"No reference images found: {reference_images_dir}")
    return names


def _target_style(name: str) -> str:
    return Path(name).stem.split("_to_", 1)[-1]


def migrate_one(name: str, source_dir: Path, output_root: Path, reference_names: set[str]) -> dict[str, object]:
    t0 = time.time()
    if not source_dir.is_dir():
        return {
            "baseline": name,
            "source_dir": str(source_dir),
            "output_root": str(output_root),
            "status": "missing_source",
            "copied": 0,
            "missing": len(reference_names),
            "elapsed_sec": 0.0,
        }

    source_by_name = {p.name: p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    missing = sorted(reference_names - set(source_by_name))
    copied = 0
    if not missing:
        for file_name in sorted(reference_names):
            tgt = _target_style(file_name)
            dst = output_root / tgt / file_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_by_name[file_name], dst)
            copied += 1

        images_dir = output_root / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)
        for file_name in sorted(reference_names):
            shutil.copy2(output_root / _target_style(file_name) / file_name, images_dir / file_name)

    return {
        "baseline": name,
        "source_dir": str(source_dir),
        "output_root": str(output_root),
        "status": "ok" if not missing else "incomplete",
        "copied": copied,
        "missing": len(missing),
        "missing_preview": ";".join(missing[:10]),
        "elapsed_sec": round(time.time() - t0, 3),
    }


def main() -> int:
    workspace_root = Path(__file__).resolve().parents[2]
    default_ref = (
        workspace_root
        / "SchrodingerBridge"
        / "exp"
        / "pareto_probe_4"
        / "S-add__K-3_C-2_W-10_Col-15"
        / "full_eval"
        / "epoch_0001"
        / "images"
    )
    parser = argparse.ArgumentParser(description="Manually migrate existing run outputs into protocol_a_800 folders.")
    parser.add_argument("--reference-images-dir", type=Path, default=default_ref)
    parser.add_argument("--protocol", default="protocol_a_800")
    parser.add_argument("--baselines", nargs="+", default=list(DEFAULT_SOURCES))
    args = parser.parse_args()

    reference_names = _reference_names(args.reference_images_dir.resolve())
    rows: list[dict[str, object]] = []
    for baseline in args.baselines:
        if baseline not in DEFAULT_SOURCES:
            rows.append({"baseline": baseline, "status": "unknown_source", "copied": 0, "missing": len(reference_names)})
            continue
        source_dir = (workspace_root / DEFAULT_SOURCES[baseline]).resolve()
        output_root = (workspace_root / "Related_Works" / "baseline_pipeline" / "results" / baseline / args.protocol).resolve()
        rows.append(migrate_one(baseline, source_dir, output_root, reference_names))

    out_dir = workspace_root / "Related_Works" / "baseline_pipeline" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"manual_migration_{args.protocol}.json"
    csv_path = out_dir / f"manual_migration_{args.protocol}.csv"
    json_path.write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["baseline", "source_dir", "output_root", "status", "copied", "missing", "missing_preview", "elapsed_sec"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})

    print(f"migration json: {json_path}")
    print(f"migration csv : {csv_path}")
    for row in rows:
        print(f"{row.get('baseline')}: {row.get('status')} copied={row.get('copied')} missing={row.get('missing')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
