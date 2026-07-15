from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "experiments"
AAAI = ROOT / "aaai2027"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _fmt(value: str) -> str:
    text = str(value or "").strip()
    try:
        return f"{float(text):.4f}"
    except Exception:
        return text


def _md_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> list[str]:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        vals = []
        for key, _ in columns:
            vals.append(_fmt(row.get(key, "")))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> int:
    results_master = _read_csv(DOCS / "aaai2027_results_master.csv")
    inmortal_master = _read_csv(DOCS / "aaai2027_inmortal_results_master.csv")
    same_cost = _read_csv(DOCS / "2026-06-04-distinct5_same_cost_inventory.csv")
    nonclip_board = _read_csv(AAAI / "current_mainline_evidence_board_20260609.csv")
    vlm_board = _read_csv(AAAI / "vlm_lbmpsv2_vs_seedream_vs_samst_vs_samam_20260610_snapshot6.method_summary.csv")

    out = DOCS / "2026-06-10-complete-experiment-catalog.md"
    lines: list[str] = []
    lines.append("# Complete Experiment Catalog")
    lines.append("")
    lines.append("Date: 2026-06-10")
    lines.append("")
    lines.append("This document is the flat all-data catalog for the current phase.")
    lines.append("It is generated from the authoritative CSV registries rather than handwritten.")
    lines.append("")

    lines.append("## 1. Paper-Facing Mixed Registry")
    lines.append("")
    lines.extend(
        _md_table(
            results_master,
            [
                ("experiment", "Experiment"),
                ("dataset", "Dataset"),
                ("method", "Method"),
                ("variant", "Variant"),
                ("selection", "Selection"),
                ("train_batch", "Batch"),
                ("train_epochs", "Epochs"),
                ("clip_style", "TransferStyle"),
                ("content_lpips", "TransferLPIPS"),
                ("full_clip_style", "FullStyle"),
                ("full_content_lpips", "FullLPIPS"),
                ("delta_idt_transfer", "DeltaIDT"),
                ("train_wall", "TrainWall"),
                ("status", "Status"),
                ("decision", "Decision"),
                ("evidence_path", "Evidence"),
            ],
        )
    )
    lines.append("")

    lines.append("## 2. Inmortal Mechanism Registry")
    lines.append("")
    lines.extend(
        _md_table(
            inmortal_master,
            [
                ("experiment", "Experiment"),
                ("family", "Family"),
                ("reading", "Reading"),
                ("selection", "Selection"),
                ("train_batch", "Batch"),
                ("train_epochs", "Epochs"),
                ("transfer_clip_style", "TransferStyle"),
                ("transfer_content_lpips", "TransferLPIPS"),
                ("full_clip_style", "FullStyle"),
                ("full_content_lpips", "FullLPIPS"),
                ("status", "Status"),
                ("evidence_path", "Evidence"),
            ],
        )
    )
    lines.append("")

    lines.append("## 3. Distinct5 Same-Cost / Convergence Inventory")
    lines.append("")
    lines.extend(
        _md_table(
            same_cost,
            [
                ("dataset", "Dataset"),
                ("method", "Method"),
                ("arm", "Arm"),
                ("status", "Status"),
                ("transfer_clip_style", "TransferStyle"),
                ("transfer_lpips", "TransferLPIPS"),
                ("delta_idt_transfer", "DeltaIDT"),
                ("full_clip_style", "FullStyle"),
                ("full_lpips", "FullLPIPS"),
                ("note", "Note"),
            ],
        )
    )
    lines.append("")

    lines.append("## 4. Non-CLIP Mainline Evidence Board")
    lines.append("")
    lines.extend(
        _md_table(
            nonclip_board,
            [
                ("label", "Label"),
                ("run", "Run"),
                ("introstyle_target_score", "IntroTarget"),
                ("introstyle_delta_idt", "DeltaIDT"),
                ("introstyle_margin", "Margin"),
                ("dino_structure", "DINO"),
                ("vlm_cases_completed", "VLMCases"),
                ("vlm_wins", "VLMWins"),
                ("current_read", "Read"),
                ("evidence_path", "Evidence"),
            ],
        )
    )
    lines.append("")

    lines.append("## 5. Current Four-Way VLM Board")
    lines.append("")
    lines.extend(
        _md_table(
            vlm_board,
            [
                ("method", "Method"),
                ("cases_completed", "Cases"),
                ("wins_so_far", "Wins"),
                ("win_rate_so_far", "WinRate"),
                ("style_wins_so_far", "StyleWins"),
                ("structure_wins_so_far", "StructWins"),
                ("artifact_wins_so_far", "ArtifactWins"),
                ("mean_style_specificity", "MeanStyle"),
                ("mean_structure_preservation", "MeanStruct"),
                ("mean_artifact_control", "MeanArtifact"),
            ],
        )
    )
    lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
