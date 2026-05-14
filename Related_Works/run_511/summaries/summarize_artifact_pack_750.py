"""Summarize stronger artifact-pack diagnostics for complete_750 runs."""
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
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return next((row for row in data.get("results", []) if row.get("target") == "ALL"), {})


def to_float(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def risk_flags(row: dict[str, object]) -> str:
    flags: list[str] = []
    if to_float(row.get("musiq")) < 48:
        flags.append("low_nr_quality")
    if to_float(row.get("maniqa")) < 0.40:
        flags.append("low_maniqa")
    if to_float(row.get("dists_content")) > 0.20:
        flags.append("content_distance_high")
    if to_float(row.get("denoise_style_drop")) > 0.01:
        flags.append("denoise_fragile_style")
    if to_float(row.get("fft_radial_kl_style")) > 0.20:
        flags.append("hf_distribution_shift")
    if to_float(row.get("chroma_grain_index")) > 0.60:
        flags.append("chroma_grain")
    return ",".join(flags)


def build_rows() -> list[dict[str, object]]:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest["runs"]:
        run = item["run"]
        art = load_overall(ROOT / run / "eval_artifact_pack750.json")
        if not art:
            continue
        row = {
            "run": run,
            "method": DISPLAY.get(run, run),
            "images": item["images"],
            "musiq": art.get("musiq", ""),
            "maniqa": art.get("maniqa", ""),
            "dists_content": art.get("dists_content", ""),
            "denoise_style_drop": art.get("denoise_style_drop", ""),
            "denoise_chroma_delta": art.get("denoise_chroma_delta", ""),
            "fft_radial_kl_style": art.get("fft_radial_kl_style", ""),
            "fft_slope_error": art.get("fft_slope_error", ""),
            "chroma_acl_z": art.get("chroma_acl_z", ""),
            "chroma_moran_z": art.get("chroma_moran_z", ""),
            "small_blob_ratio_z": art.get("small_blob_ratio_z", ""),
            "chroma_grain_index": art.get("chroma_grain_index", ""),
        }
        row["risk_flags"] = risk_flags(row)
        rows.append(row)
    return rows


def write_reports(rows: list[dict[str, object]], stem: str, title: str) -> None:
    if not rows:
        return
    csv_path = ROOT / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = [
        f"# {title}",
        "",
        "Source folder: `run_511/complete_750`",
        "",
        "| Method | Run | MUSIQ up | MANIQA up | DISTS-content down | DenoiseStyleDrop down | FFT-KL down | ACL-Z | Moran-Z | Blob-Z | GrainIndex down | Risk flags |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        md.append(
            f"| {row['method']} | `{row['run']}` | {row['musiq']} | {row['maniqa']} | "
            f"{row['dists_content']} | {row['denoise_style_drop']} | {row['fft_radial_kl_style']} | "
            f"{row['chroma_acl_z']} | {row['chroma_moran_z']} | {row['small_blob_ratio_z']} | "
            f"{row['chroma_grain_index']} | {row['risk_flags']} |"
        )
    md.extend(
        [
            "",
            "## Notes",
            "",
            "- `MUSIQ` and `MANIQA` are no-reference quality metrics; higher is better.",
            "- `DISTS-content` is computed against the source content image; lower is better.",
            "- `DenoiseStyleDrop` measures how much CLIP-style falls after mild bilateral denoising.",
            "- `GrainIndex` combines short chroma autocorrelation, weak chroma spatial coherence, and excess small chroma blobs.",
        ]
    )
    md_path = ROOT / f"{stem}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


def main() -> int:
    rows = build_rows()
    write_reports(rows, "summary_artifact_pack_750", "Complete 750 Artifact-Pack Summary")
    related_rows = [row for row in rows if row["run"] != "ours_epoch_0007"]
    write_reports(related_rows, "summary_artifact_related_works_750", "Related Works 750 Artifact-Pack Summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
