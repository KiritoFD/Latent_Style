"""Build a compact report for current protocol-750 SB-match screening results."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
RUN511_ROOT = THIS_DIR.parent
WORKSPACE_ROOT = RUN511_ROOT.parent.parent
REFERENCE_IMAGES = (
    WORKSPACE_ROOT
    / "SchrodingerBridge"
    / "exp"
    / "pareto_probe_4"
    / "S-add__K-3_C-2_W-10_Col-15"
    / "full_eval"
    / "epoch_0001"
    / "images"
)

RUNS = [
    (
        "Ours epoch_0007",
        "ours_k1_c0_w20_col0_epoch_0007",
        WORKSPACE_ROOT
        / "SchrodingerBridge"
        / "S-add__K-1_C-0_W-20_Col-0"
        / "full_eval"
        / "epoch_0007"
        / "images",
        RUN511_ROOT / "outputs" / "ours_k1_c0_w20_col0" / "epoch_0007" / "eval_protocol750_sbmatch.json",
        "ok",
    ),
    (
        "StyleID strict",
        "styleid_750_strict",
        RUN511_ROOT / "outputs" / "styleid_750_strict" / "infer_750" / "images",
        RUN511_ROOT / "outputs" / "styleid_750_strict" / "infer_750" / "eval_protocol750_sbmatch.json",
        "ok",
    ),
    (
        "SaMST strict",
        "samst_750_strict",
        RUN511_ROOT / "outputs" / "samst_750_strict" / "infer_750" / "images",
        RUN511_ROOT / "outputs" / "samst_750_strict" / "infer_750" / "eval_protocol750_sbmatch.json",
        "ok",
    ),
    (
        "AdaIN v32k",
        "adain_7g_v32k",
        RUN511_ROOT / "outputs" / "adain_7g_v32k" / "infer_750" / "images",
        RUN511_ROOT / "outputs" / "adain_7g_v32k" / "infer_750" / "eval_protocol750_sbmatch.json",
        "ok",
    ),
    (
        "AdaIN vgg19",
        "adain_7g_vgg19",
        RUN511_ROOT / "outputs" / "adain_7g_vgg19" / "infer_750" / "images",
        RUN511_ROOT / "outputs" / "adain_7g_vgg19" / "infer_750" / "eval_protocol750_sbmatch.json",
        "ok",
    ),
    (
        "AdaIN bad",
        "adain_4g_real",
        RUN511_ROOT / "outputs" / "adain_4g_real" / "infer_750" / "images",
        RUN511_ROOT / "outputs" / "adain_4g_real" / "infer_750" / "eval_protocol750_sbmatch.json",
        "invalid",
    ),
    (
        "SaMST refmatch",
        "samst_750_refmatch",
        RUN511_ROOT / "outputs" / "samst_750_refmatch" / "infer_750" / "images",
        RUN511_ROOT / "outputs" / "samst_750_refmatch" / "infer_750" / "eval_protocol750_sbmatch.json",
        "partial",
    ),
    (
        "StyleID refmatch",
        "styleid_750_refmatch",
        RUN511_ROOT / "outputs" / "styleid_750_refmatch" / "infer_750" / "images",
        RUN511_ROOT / "outputs" / "styleid_750_refmatch" / "infer_750" / "eval_protocol750_sbmatch.json",
        "partial",
    ),
]

GUARD_PATHS = {
    "ours_k1_c0_w20_col0_epoch_0007": RUN511_ROOT / "outputs" / "ours_k1_c0_w20_col0" / "epoch_0007" / "eval_guard750.json",
    "styleid_750_strict": RUN511_ROOT / "outputs" / "styleid_750_strict" / "infer_750" / "eval_guard750.json",
    "samst_750_strict": RUN511_ROOT / "outputs" / "samst_750_strict" / "infer_750" / "eval_guard750.json",
    "adain_7g_v32k": RUN511_ROOT / "outputs" / "adain_7g_v32k" / "infer_750" / "eval_guard750.json",
}


def load_overall(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return next((row for row in data.get("results", []) if row.get("target") == "ALL"), {})


def load_guard(run: str) -> dict[str, object]:
    path = GUARD_PATHS.get(run)
    if path is None or not path.exists():
        return {}
    return load_overall(path)


def main() -> int:
    ref_names = {p.name for p in REFERENCE_IMAGES.glob("*.jpg")}
    rows: list[dict[str, object]] = []

    for method, run, images_dir, eval_path, status in RUNS:
        names = {p.name for p in images_dir.glob("*.jpg")} if images_dir.exists() else set()
        targets = Counter(
            name.rsplit("_to_", 1)[-1].removesuffix(".jpg")
            for name in names
            if "_to_" in name
        )
        overall = load_overall(eval_path)
        guard = load_guard(run)
        rows.append(
            {
                "method": method,
                "run": run,
                "status": status,
                "images": len(names),
                "match_ref": len(names & ref_names),
                "missing_ref": len(ref_names - names),
                "extra": len(names - ref_names),
                "targets": " ".join(f"{k}:{targets[k]}" for k in sorted(targets)),
                "lpips": overall.get("lpips", ""),
                "clip_style": overall.get("clip_style", ""),
                "clip_content": overall.get("clip_content", ""),
                "ssim_y": guard.get("ssim_y", ""),
                "edge_f1": guard.get("edge_f1", ""),
                "hf_ratio_src": guard.get("hf_ratio_src", ""),
            }
        )

    docs_dir = RUN511_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = docs_dir / "protocol750_eval_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Protocol-750 Evaluation Report",
        "",
        "Reference manifest: `SchrodingerBridge/exp/pareto_probe_4/"
        "S-add__K-3_C-2_W-10_Col-15/full_eval/epoch_0001/images`",
        "",
        "Metric protocol: SB-match (`CLIP-style = cos(CLIP(gen), mean target-style reference prototype)`, `LPIPS = VGG-LPIPS`).",
        "",
        "| Method | Run | Status | Images | Ref match | Missing | Extra | LPIPS down | CLIP-style up | CLIP-content up | SSIM-Y up | Edge-F1 up | HF ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['method']} | `{row['run']}` | {row['status']} | "
            f"{row['images']} | {row['match_ref']} | {row['missing_ref']} | {row['extra']} | "
            f"{row['lpips']} | {row['clip_style']} | {row['clip_content']} | "
            f"{row['ssim_y']} | {row['edge_f1']} | {row['hf_ratio_src']} |"
        )
    md.extend(
        [
            "",
            "## Notes",
            "",
            "- `ok` rows exactly match the 750-image manifest and are table-ready for current screening metrics.",
            "- `partial` rows are evaluated only on files that overlap the manifest, so they are useful for diagnosis but not main-table ready.",
            "- `invalid` rows have exact 750 files but clearly broken/too weak behavior and should not be used.",
            "- Guard metrics are no-download sanity checks. They do not replace DINO/CFSD/user study, but help catch CLIP/LPIPS blind spots.",
            "- SaMST has strong structure metrics, but visual inspection shows heavy pointillist/grain artifacts; keep qualitative grids and user study in the decision loop.",
        ]
    )

    md_path = docs_dir / "protocol750_eval_report.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
