from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent

INTROSTYLE_CSV = ROOT / "introstyle_page1" / "introstyle_page1_summary.csv"
NONCLIP_CSV = ROOT / "distinct5_nonclip_style_probe.csv"
RESULTS_MASTER_CSV = WORKSPACE / "SchrodingerBridge" / "docs" / "experiments" / "aaai2027_results_master.csv"
BEST_CSV = WORKSPACE / "best.csv"
SELECTED_METRICS_CSV = ROOT / "distinct5_operating_point_selected_style_metrics.csv"
BOOTSTRAP_CSV = ROOT / "distinct5_idt_bootstrap_extended.csv"

OUT_CSV = ROOT / "introstyle_page1" / "page1_shortlist_comparison.csv"
OUT_MD = ROOT / "introstyle_page1" / "page1_shortlist_comparison.md"


POINTS = [
    {"run": "IDT", "label": "IDT", "clip_source": ("results_master", "distinct5_512__idt__no_op"), "nonclip_run": "IDT"},
    {"run": "SaMAM_2250", "label": "SaMAM-2250", "clip_source": ("results_master", "distinct5_512__SaMAM__step_2250")},
    {"run": "SaMST_e15", "label": "SaMST e15", "clip_source": ("results_master", "distinct5_512__SaMST__e15"), "nonclip_run": "SaMST_e15"},
    {"run": "Lat_SaMAM_step1500", "label": "Lat SaMAM", "clip_source": ("results_master", "distinct5_512__SaMAM-latent__convergence__step1500_fast")},
    {"run": "Lat_SaMST_batch1050", "label": "Lat SaMST", "clip_source": ("results_master", "distinct5_512__SaMST-latent__convergence__batch1050_fast")},
    {"run": "LBM-K_e1", "label": "LBM-K", "clip_source": ("best", "best_compact_mainline_anchor"), "nonclip_run": "LBM-K_e1"},
    {"run": "LBM-Knee_e13", "label": "LBM-Knee", "clip_source": ("best", "best_promoted_lpips_ge_070"), "nonclip_run": "LBM-Knee_e13"},
    {"run": "LBM-PS-v2_e13", "label": "LBM-PS-v2", "clip_source": ("best", "style_best_current"), "nonclip_run": "LBM-PS-v2_e13"},
    {"run": "Seedream_repaired750", "label": "Seedream-4.5", "clip_source": ("selected_metrics", "Seedream_repaired750"), "nonclip_run": "Seedream_repaired750"},
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_introstyle() -> dict[str, dict[str, str]]:
    return {row["run"]: row for row in read_csv(INTROSTYLE_CSV)}


def read_nonclip() -> dict[str, dict[str, str]]:
    return {row["run"]: row for row in read_csv(NONCLIP_CSV)}


def read_results_master() -> dict[str, dict[str, str]]:
    return {row["experiment"]: row for row in read_csv(RESULTS_MASTER_CSV)}


def read_best() -> dict[str, dict[str, str]]:
    return {row["slot"]: row for row in read_csv(BEST_CSV)}


def read_selected_metrics() -> dict[str, dict[str, str]]:
    return {row["run"]: row for row in read_csv(SELECTED_METRICS_CSV)}


def read_bootstrap() -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in read_csv(BOOTSTRAP_CSV) if row["scope"] == "transfer"}


def fmt(x: str | float | None) -> str:
    if x is None or x == "":
        return ""
    return f"{float(x):.4f}"


def main() -> int:
    intro = read_introstyle()
    nonclip = read_nonclip()
    results_master = read_results_master()
    best = read_best()
    selected = read_selected_metrics()
    bootstrap = read_bootstrap()

    rows: list[dict[str, str]] = []
    for point in POINTS:
        run = point["run"]
        intro_row = intro[run]

        clip_source_kind, clip_source_key = point["clip_source"]
        clip_style = ""
        content_lpips = ""
        delta_idt_tr = ""
        if clip_source_kind == "results_master":
            row = results_master[clip_source_key]
            clip_style = row["clip_style"]
            content_lpips = row["content_lpips"]
            delta_idt_tr = row.get("delta_idt_transfer", "")
        elif clip_source_kind == "best":
            row = best[clip_source_key]
            clip_style = row["clip_style"]
            content_lpips = row["content_lpips"]
            delta_idt_tr = row["delta_idt_tr"]
        elif clip_source_kind == "selected_metrics":
            row = selected[clip_source_key]
            clip_style = row["clip_style_up"]
            content_lpips = row["lpips_down"]
            delta_idt_tr = bootstrap["Seedream-4.5"]["delta_idt_clip_style"]
        else:
            raise ValueError(clip_source_kind)

        nonclip_row = nonclip.get(point.get("nonclip_run", ""))
        rows.append(
            {
                "label": point["label"],
                "run": run,
                "transfer_clip_style": fmt(clip_style),
                "transfer_content_lpips": fmt(content_lpips),
                "transfer_delta_idt_clip": fmt(delta_idt_tr),
                "introstyle_target_score_smoke20": fmt(intro_row["transfer_target_style_score"]),
                "introstyle_delta_idt_smoke20": fmt(intro_row["transfer_delta_idt_style"]),
                "introstyle_margin_smoke20": fmt(intro_row["transfer_style_margin"]),
                "nonclip_transfer_target_acc": fmt(nonclip_row["transfer_target_acc"]) if nonclip_row else "",
                "nonclip_transfer_margin": fmt(nonclip_row["transfer_target_source_margin"]) if nonclip_row else "",
            }
        )

    fieldnames = list(rows[0].keys())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Page-1 Shortlist Comparison",
        "",
        "| Label | CLIP-style | LPIPS | Delta-IDT CLIP | IntroStyle target | IntroStyle delta-IDT | IntroStyle margin | Non-CLIP acc | Non-CLIP margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['label']} | {row['transfer_clip_style']} | {row['transfer_content_lpips']} | "
            f"{row['transfer_delta_idt_clip']} | {row['introstyle_target_score_smoke20']} | "
            f"{row['introstyle_delta_idt_smoke20']} | {row['introstyle_margin_smoke20']} | "
            f"{row['nonclip_transfer_target_acc']} | {row['nonclip_transfer_margin']} |"
        )
    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(OUT_CSV)
    print(OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
