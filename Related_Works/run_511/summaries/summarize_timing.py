"""Summarize available train/infer timing from run_511 outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
RUN511_ROOT = THIS_DIR.parent
ROOT = RUN511_ROOT / "outputs"


def load_summary(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_stage(data: dict[str, object], stage: str) -> dict[str, object] | None:
    for item in data.get("runs", []):
        if item.get("stage") == stage:
            return item
    return None


def status_of(stage: dict[str, object] | None) -> str:
    if not stage:
        return ""
    return str(stage.get("status", ""))


def fnum(value: object, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def main() -> int:
    notes = {
        "styleid_750_strict": "Inference timing is not a fair full-750 run: `photo` was actually generated (~603s), other targets were reused/copied.",
        "samst_750_strict": "Only strict full-750 inference time is recorded here; training time was not preserved in `summary.json`.",
        "stytr2_smoke6": "Training is a 1-iter smoke run (`max_iter=1`), not a full epoch.",
        "cast_smoke3": "Training is a 1-epoch smoke run; inference failed in this smoke config.",
        "adain_4g_real": "Inference succeeded but visual output is invalid; timing still recorded.",
        "adain_7g_v32k": "Full train + strict 750 inference timing available.",
        "adain_7g_vgg19": "Full train + strict 750 inference timing available.",
    }

    rows: list[dict[str, object]] = []
    for summary_path in sorted(ROOT.glob("*/summary.json")):
        run = summary_path.parent.name
        data = load_summary(summary_path)
        if not data:
            continue
        train = find_stage(data, "train")
        infer = find_stage(data, "infer")
        infer_images = infer.get("images", "") if infer else ""
        infer_elapsed = infer.get("elapsed_sec", "") if infer else ""
        per_image = ""
        if infer and infer.get("images"):
            try:
                per_image = float(infer["elapsed_sec"]) / float(infer["images"])
            except Exception:
                per_image = ""
        rows.append(
            {
                "run": run,
                "train_status": status_of(train),
                "train_elapsed_sec": train.get("elapsed_sec", "") if train else "",
                "train_batch_size": train.get("batch_size", "") if train else "",
                "train_max_iter": train.get("max_iter", "") if train else "",
                "train_n_epochs": train.get("n_epochs", "") if train else "",
                "infer_status": status_of(infer),
                "infer_elapsed_sec": infer_elapsed,
                "infer_images": infer_images,
                "infer_sec_per_image": per_image,
                "note": notes.get(run, ""),
            }
        )

    docs_dir = RUN511_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = docs_dir / "timing_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Timing Summary",
        "",
        "Source: `run_511/outputs/*/summary.json`",
        "",
        "| Run | Train | Train sec | Batch | Max iter | Epochs | Infer | Infer sec | Images | Sec / image | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        md.append(
            f"| `{row['run']}` | {row['train_status']} | {fnum(row['train_elapsed_sec'])} | "
            f"{row['train_batch_size']} | {row['train_max_iter']} | {row['train_n_epochs']} | "
            f"{row['infer_status']} | {fnum(row['infer_elapsed_sec'])} | {row['infer_images']} | "
            f"{fnum(row['infer_sec_per_image'], 6)} | {row['note']} |"
        )

    md.extend(
        [
            "",
            "## Readable Highlights",
            "",
            "- `adain_7g_v32k`: train `9220.393s`, infer `9.281s / 750 = 0.012375s per image`.",
            "- `adain_7g_vgg19`: train `262.780s`, infer `9.098s / 750 = 0.012131s per image`.",
            "- `samst_750_strict`: infer `39.826s / 750 = 0.053101s per image`; train time not preserved in current summary file.",
            "- `styleid_750_strict`: recorded infer `603.316s`, but this is not a fair full-750 timing because only `photo` was actually generated in this strict run.",
            "- `stytr2_smoke6`: smoke train `59.250s`, smoke infer `35.810s / 5 = 7.162000s per image`.",
            "- `cast_smoke3`: smoke train `29.366s` for `1` epoch, infer failed in this config.",
        ]
    )

    md_path = docs_dir / "timing_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
