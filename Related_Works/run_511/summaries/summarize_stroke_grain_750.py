"""Combine artifact-pack and HF-patch-KID into a stroke-grain summary."""
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


def build_rows() -> list[dict[str, object]]:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    raw_rows: list[dict[str, object]] = []
    for item in manifest["runs"]:
        run = item["run"]
        art = load_overall(ROOT / run / "eval_artifact_pack750.json")
        kid = load_overall(ROOT / run / "eval_hf_patch_kid750.json")
        if not art or not kid:
            continue
        raw_rows.append(
            {
                "run": run,
                "method": DISPLAY.get(run, run),
                "musiq": to_float(art.get("musiq")),
                "maniqa": to_float(art.get("maniqa")),
                "dists_content": to_float(art.get("dists_content")),
                "fft_slope_error": to_float(art.get("fft_slope_error")),
                "chroma_acl_z": to_float(art.get("chroma_acl_z")),
                "small_blob_ratio_z": to_float(art.get("small_blob_ratio_z")),
                "structure_tensor_coherence_z": to_float(art.get("structure_tensor_coherence_z")),
                "hf_patch_kid": to_float(kid.get("hf_patch_kid")),
            }
        )

    def zmap(key: str, invert: bool = False) -> dict[str, float]:
        vals = [row[key] for row in raw_rows]
        mean = sum(vals) / len(vals)
        std = max((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5, 1e-6)
        return {row["run"]: ((mean - row[key]) if invert else (row[key] - mean)) / std for row in raw_rows}

    blob_z = zmap("small_blob_ratio_z")
    acl_z = zmap("chroma_acl_z", invert=True)
    coh_z = zmap("structure_tensor_coherence_z", invert=True)
    kid_z = zmap("hf_patch_kid")
    slope_z = zmap("fft_slope_error")

    rows = []
    for row in raw_rows:
        score = (blob_z[row["run"]] + acl_z[row["run"]] + coh_z[row["run"]] + kid_z[row["run"]] + slope_z[row["run"]]) / 5.0
        row["stroke_grain_artifact"] = round(score, 4)
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
        "| Method | Run | MUSIQ up | MANIQA up | DISTS-content down | HF-Patch-KID down | FFT-slope-error down | Blob-Z down | ACL-Z | Coherence-Z | Stroke-Grain Artifact down |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['method']} | `{row['run']}` | {row['musiq']:.4f} | {row['maniqa']:.4f} | "
            f"{row['dists_content']:.4f} | {row['hf_patch_kid']:.6f} | {row['fft_slope_error']:.4f} | "
            f"{row['small_blob_ratio_z']:.4f} | {row['chroma_acl_z']:.4f} | {row['structure_tensor_coherence_z']:.4f} | "
            f"{row['stroke_grain_artifact']:.4f} |"
        )
    md_path = ROOT / f"{stem}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


def main() -> int:
    rows = build_rows()
    write_reports(rows, "summary_stroke_grain_750", "Complete 750 Stroke-Grain Summary")
    related = [row for row in rows if row["run"] != "ours_epoch_0007"]
    write_reports(related, "summary_stroke_grain_related_works_750", "Related Works 750 Stroke-Grain Summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
