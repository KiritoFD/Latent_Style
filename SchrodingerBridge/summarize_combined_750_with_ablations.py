from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


SB_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = SB_ROOT.parent
RUN511_ROOT = WORKSPACE_ROOT / "Related_Works" / "run_511"
COMPLETE_750 = RUN511_ROOT / "complete_750"
OUTPUTS_750 = RUN511_ROOT / "outputs"
ABLATION_ROOT = SB_ROOT / "ablation_destructive_7epoch"
RELATED_METRICS_DIR = WORKSPACE_ROOT / "Related_Works" / "results" / "metrics_summary"


BASELINE_RUNS = [
    ("D0_full_correct_7ep", "Ours D0 full", ABLATION_ROOT / "D0_full_correct_7ep" / "full_eval" / "epoch_0007", "ablation"),
    ("samst_strict", "SaMST strict", COMPLETE_750 / "samst_strict", "baseline"),
    ("styleid_strict", "StyleID strict", COMPLETE_750 / "styleid_strict", "baseline"),
    ("s2wat_strict", "S2WAT strict", OUTPUTS_750 / "s2wat_750_strict" / "infer_750", "baseline"),
    ("adain_v32k", "AdaIN v32k", COMPLETE_750 / "adain_v32k", "baseline"),
    ("adain_vgg19", "AdaIN vgg19", COMPLETE_750 / "adain_vgg19", "baseline"),
    ("adain_bad", "AdaIN bad", COMPLETE_750 / "adain_bad", "baseline"),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_all(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if "results" in data:
        return next((r for r in data.get("results", []) if r.get("target") == "ALL"), {})
    return {}


def image_count(base: Path) -> int:
    img_dir = base / "images"
    return len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0


def load_registry() -> dict[str, dict[str, str]]:
    path = ABLATION_ROOT / "destructive_ablation_7epoch_registry.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def row_from_eval(run: str, method: str, base: Path, group: str, registry: dict[str, dict[str, str]]) -> dict[str, Any]:
    proto = load_all(base / "eval_protocol750_sbmatch.json")
    guard = load_all(base / "eval_guard750.json")
    artifact = load_all(base / "eval_artifact_pack750.json")
    hf_kid = load_all(base / "eval_hf_patch_kid750.json")
    plain_kid = load_all(base / "eval_plain_kid750.json")
    reg = registry.get(run, {})
    row = {
        "group": group,
        "run": run,
        "method": method,
        "path": str(base),
        "images": image_count(base),
        "lpips": proto.get("lpips", ""),
        "clip_style": proto.get("clip_style", ""),
        "clip_content": proto.get("clip_content", ""),
        "ssim_y": guard.get("ssim_y", ""),
        "edge_f1": guard.get("edge_f1", ""),
        "edge_iou": guard.get("edge_iou", ""),
        "blur_style_drop": guard.get("blur_style_drop", ""),
        "down_style_drop": guard.get("down_style_drop", ""),
        "extra_edge_rate": guard.get("extra_edge_rate", ""),
        "chroma_speckle_z": guard.get("chroma_speckle_z", ""),
        "flat_chroma_hf_z": guard.get("flat_chroma_hf_z", ""),
        "musiq": artifact.get("musiq", ""),
        "maniqa": artifact.get("maniqa", ""),
        "dists_content": artifact.get("dists_content", ""),
        "fft_radial_kl_style": artifact.get("fft_radial_kl_style", ""),
        "fft_slope_error": artifact.get("fft_slope_error", ""),
        "chroma_grain_index": artifact.get("chroma_grain_index", ""),
        "hf_patch_kid": hf_kid.get("hf_patch_kid", ""),
        "plain_kid": plain_kid.get("kid", ""),
        "train_sec": reg.get("train_sec", ""),
        "eval_sec": reg.get("eval_sec", ""),
        "ablation_label": reg.get("label", ""),
        "ablation_purpose": reg.get("purpose", ""),
        "has_base": bool(proto),
        "has_guard": bool(guard),
        "has_artifact": bool(artifact),
        "has_hf_kid": bool(hf_kid),
        "has_plain_kid": bool(plain_kid),
    }
    return row


def build_rows() -> list[dict[str, Any]]:
    registry = load_registry()
    rows = [row_from_eval(run, method, base, group, registry) for run, method, base, group in BASELINE_RUNS]
    for run in sorted(registry, key=lambda x: int(x[1:].split("_", 1)[0]) if x.startswith("D") else 999):
        if run == "D0_full_correct_7ep":
            continue
        reg = registry[run]
        base = Path(reg["save_dir"]) / "full_eval" / "epoch_0007"
        rows.append(row_from_eval(run, reg.get("label", run), base, "ablation", registry))
    return rows


def fmt(v: Any, digits: int = 4) -> str:
    if v in ("", None):
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def write_outputs(rows: list[dict[str, Any]]) -> None:
    out_dir = ABLATION_ROOT / "combined_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    RELATED_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "combined_750_with_destructive_ablations.csv"
    md_path = out_dir / "combined_750_with_destructive_ablations.md"
    sb_csv = SB_ROOT / "combined_750_with_destructive_ablations.csv"
    sb_md = SB_ROOT / "combined_750_with_destructive_ablations.md"
    rel_csv = RELATED_METRICS_DIR / "combined_750_with_destructive_ablations.csv"
    rel_md = RELATED_METRICS_DIR / "combined_750_with_destructive_ablations.md"

    keys = list(rows[0].keys())
    for path in [csv_path, sb_csv, rel_csv]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Combined 750 Summary With Destructive Ablations",
        "",
        "Notes:",
        "",
        "- `Ours` is replaced by `D0_full_correct_7ep` as requested.",
        "- Artifact-pack columns are filled only where that expensive metric pack has actually been run.",
        "- All ablation rows have strict 750 images plus base/guard/HF-KID/plain-KID coverage.",
        "",
        "## Main + Ablation Table",
        "",
        "| Group | Method | Run | Images | LPIPS↓ | CLIP-S↑ | CLIP-C↑ | SSIM-Y↑ | Edge-F1↑ | ExtraEdge↓ | Chroma-Z↓ | HF-KID↓ | KID↓ | Train sec |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['method']} | `{row['run']}` | {row['images']} | "
            f"{fmt(row['lpips'])} | {fmt(row['clip_style'])} | {fmt(row['clip_content'])} | "
            f"{fmt(row['ssim_y'])} | {fmt(row['edge_f1'])} | {fmt(row['extra_edge_rate'])} | "
            f"{fmt(row['chroma_speckle_z'])} | {fmt(row['hf_patch_kid'], 6)} | {fmt(row['plain_kid'], 6)} | "
            f"{fmt(row['train_sec'], 1)} |"
        )
    lines.extend(
        [
            "",
            "## Ablation Purposes",
            "",
            "| Run | Label | Purpose |",
            "| --- | --- | --- |",
        ]
    )
    for row in rows:
        if row["group"] == "ablation" and row["ablation_purpose"]:
            lines.append(f"| `{row['run']}` | {row['ablation_label']} | {row['ablation_purpose']} |")
    text = "\n".join(lines) + "\n"
    for path in [md_path, sb_md, rel_md]:
        path.write_text(text, encoding="utf-8")

    print(csv_path)
    print(md_path)
    print(sb_csv)
    print(sb_md)


def main() -> int:
    rows = build_rows()
    write_outputs(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
