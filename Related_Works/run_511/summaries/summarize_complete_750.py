"""Summarize complete protocol-750 evaluations into one table."""
from __future__ import annotations

import csv
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
RUN511_ROOT = THIS_DIR.parent
ROOT = RUN511_ROOT / "complete_750"

DISPLAY = {
    "ours_epoch_0007": "Ours epoch_0007",
    "samst_strict": "SaMST strict",
    "styleid_strict": "StyleID strict",
    "adain_v32k": "AdaIN v32k",
    "adain_vgg19": "AdaIN vgg19",
    "adain_bad": "AdaIN bad",
}


def load_overall(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return next((row for row in data.get("results", []) if row.get("target") == "ALL"), {})


def to_float(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def risk_flags(row: dict[str, object]) -> str:
    flags: list[str] = []
    lpips = to_float(row.get("lpips", 0.0))
    clip_content = to_float(row.get("clip_content", 0.0))
    noise_ratio_style = to_float(row.get("noise_ratio_style", 0.0))
    hf_ratio_style = to_float(row.get("hf_ratio_style", 0.0))
    edge_f1 = to_float(row.get("edge_f1", 0.0))
    blockiness = to_float(row.get("blockiness", 0.0))
    blur_style_drop = to_float(row.get("blur_style_drop", 0.0))
    down_style_drop = to_float(row.get("down_style_drop", 0.0))
    extra_edge_rate = to_float(row.get("extra_edge_rate", 0.0))
    chroma_speckle_z = to_float(row.get("chroma_speckle_z", 0.0))
    flat_chroma_hf_z = to_float(row.get("flat_chroma_hf_z", 0.0))

    if lpips > 0.72:
        flags.append("weak_content")
    if clip_content < 0.60:
        flags.append("semantic_drift")
    if noise_ratio_style > 1.08 or hf_ratio_style > 1.08:
        flags.append("noisy_vs_style")
    if edge_f1 < 0.03:
        flags.append("washed_structure")
    if blockiness > 1.45:
        flags.append("blocky")
    if blur_style_drop > 0.015:
        flags.append("style_not_blur_robust")
    if down_style_drop > 0.01:
        flags.append("style_not_scale_robust")
    if chroma_speckle_z > 1.0 or flat_chroma_hf_z > 1.0:
        flags.append("chroma_speckle")
    if extra_edge_rate > 0.45:
        flags.append("extra_edges")
    if edge_f1 > 0.45 and (noise_ratio_style > 0.88 or chroma_speckle_z > 0.8 or flat_chroma_hf_z > 0.8):
        flags.append("grainy_but_structured")
    return ",".join(flags)


def build_rows() -> list[dict[str, object]]:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest["runs"]:
        run = item["run"]
        sb = load_overall(ROOT / run / "eval_protocol750_sbmatch.json")
        guard = load_overall(ROOT / run / "eval_guard750.json")
        row = {
            "run": run,
            "method": DISPLAY.get(run, run),
            "images": item["images"],
            "lpips": sb.get("lpips", ""),
            "clip_style": sb.get("clip_style", ""),
            "clip_content": sb.get("clip_content", ""),
            "ssim_y": guard.get("ssim_y", ""),
            "edge_f1": guard.get("edge_f1", ""),
            "edge_iou": guard.get("edge_iou", ""),
            "hf_ratio_src": guard.get("hf_ratio_src", ""),
            "hf_ratio_style": guard.get("hf_ratio_style", ""),
            "tv_ratio_style": guard.get("tv_ratio_style", ""),
            "noise_ratio_style": guard.get("noise_ratio_style", ""),
            "lap_var_ratio_style": guard.get("lap_var_ratio_style", ""),
            "blockiness": guard.get("blockiness", ""),
            "blur_style_clip": guard.get("blur_style_clip", ""),
            "down_style_clip": guard.get("down_style_clip", ""),
            "blur_style_drop": guard.get("blur_style_drop", ""),
            "down_style_drop": guard.get("down_style_drop", ""),
            "extra_edge_rate": guard.get("extra_edge_rate", ""),
            "edge_density_ratio": guard.get("edge_density_ratio", ""),
            "chroma_speckle_z": guard.get("chroma_speckle_z", ""),
            "flat_chroma_hf_z": guard.get("flat_chroma_hf_z", ""),
        }
        row["risk_flags"] = risk_flags(row)
        rows.append(row)
    return rows


def write_reports(rows: list[dict[str, object]], stem: str, title: str) -> None:
    csv_path = ROOT / f"{stem}.csv"
    keys = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    md = [
        f"# {title}",
        "",
        "Source folder: `run_511/complete_750`",
        "",
        "| Method | Run | LPIPS down | CLIP-style up | Blur-drop down | Down-drop down | CLIP-content up | SSIM-Y up | Edge-F1 up | Extra-edge down | Chroma-Z down | FlatChroma-Z down | Risk flags |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        md.append(
            f"| {row['method']} | `{row['run']}` | {row['lpips']} | {row['clip_style']} | "
            f"{row['blur_style_drop']} | {row['down_style_drop']} | {row['clip_content']} | {row['ssim_y']} | "
            f"{row['edge_f1']} | {row['extra_edge_rate']} | {row['chroma_speckle_z']} | {row['flat_chroma_hf_z']} | {row['risk_flags']} |"
        )
    md.extend(
        [
            "",
            "## Notes",
            "",
            "- `Blur-drop` and `Down-drop` measure how much CLIP-style falls after mild blur or down-up sampling.",
            "- Large positive `Chroma-Z` / `FlatChroma-Z` suggests color-speckle behavior stronger than the target style distribution.",
            "- `Extra-edge` measures output edges that appear outside a dilated content-edge support mask.",
        ]
    )
    md_path = ROOT / f"{stem}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


def main() -> int:
    rows = build_rows()
    csv_path = ROOT / "summary_complete_750.csv"
    write_reports(rows, "summary_complete_750", "Complete 750 Evaluation Summary")
    related_rows = [row for row in rows if row["run"] != "ours_epoch_0007"]
    write_reports(related_rows, "summary_related_works_750", "Related Works 750 Evaluation Summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
