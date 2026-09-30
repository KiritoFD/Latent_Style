from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[2]
SB = REPO / "SchrodingerBridge"
RESULTS = REPO / "Related_Works" / "baseline_pipeline" / "results"

OUT_DIR = (
    SB
    / "docs"
    / "experiments"
    / "distinct5_512_20260602"
    / "baseline_packet_status_20260604"
)

IDT_METRICS = SB / "docs" / "experiments" / "idt_eval_20260602" / "distinct5_512" / "idt_5x5" / "metrics.csv"

SAMST = {
    "e5": {
        "metrics": RESULTS
        / "samst_distinct5_512_real_b1_e5_20260603"
        / "eval_bundle"
        / "eval_epoch5"
        / "epoch_0005"
        / "metrics.csv",
        "summary": RESULTS
        / "samst_distinct5_512_real_b1_e5_20260603"
        / "eval_bundle"
        / "eval_epoch5"
        / "epoch_0005"
        / "summary.json",
        "artfid": RESULTS
        / "samst_distinct5_512_real_b1_e5_20260603"
        / "eval_bundle"
        / "eval_epoch5"
        / "epoch_0005"
        / "aggregate_targetwise_artfid.json",
    },
    "e15": {
        "metrics": RESULTS
        / "samst_distinct5_512_real_b2_e15_20260602"
        / "eval_epoch15"
        / "epoch_0015"
        / "metrics.csv",
        "summary": RESULTS
        / "samst_distinct5_512_real_b2_e15_20260602"
        / "eval_epoch15"
        / "epoch_0015"
        / "summary.json",
        "artfid": RESULTS
        / "samst_distinct5_512_real_b2_e15_20260602"
        / "eval_epoch15"
        / "epoch_0015"
        / "aggregate_targetwise_artfid.json",
    },
}

SAMST_COMPARE = (
    RESULTS
    / "samst_distinct5_512_real_b1_e5_20260603"
    / "eval_bundle"
    / "compare_e5_vs_e15"
    / "samst_distinct5_epoch_comparison.json"
)

SAMAM_EXPECTED_REMOTE = Path(
    "I:/Github/Latent_Style/Related_Works/baseline_pipeline/results/"
    "samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag/eval_curve"
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["src_style"], row["tgt_style"], row["src_image"]


def _scope(rows: list[dict[str, str]], transfer_only: bool) -> list[dict[str, str]]:
    if not transfer_only:
        return rows
    return [r for r in rows if r["src_style"] != r["tgt_style"]]


def _avg(rows: list[dict[str, str]], field: str) -> float:
    return float(mean(float(r[field]) for r in rows))


def _align_rows(label: str, idt_rows: list[dict[str, str]], method_rows: list[dict[str, str]]) -> dict[str, object]:
    idt_by_key = {_key(r): r for r in idt_rows}
    method_by_key = {_key(r): r for r in method_rows}
    idt_keys = set(idt_by_key)
    method_keys = set(method_by_key)
    common = sorted(idt_keys & method_keys)

    aligned_rows: list[dict[str, object]] = []
    for key in common:
        idt = idt_by_key[key]
        method = method_by_key[key]
        aligned_rows.append(
            {
                "src_style": key[0],
                "tgt_style": key[1],
                "src_image": key[2],
                "scope": "identity" if key[0] == key[1] else "transfer",
                "idt_clip_style": float(idt["clip_style"]),
                "method_clip_style": float(method["clip_style"]),
                "delta_clip_style": float(method["clip_style"]) - float(idt["clip_style"]),
                "idt_content_lpips": float(idt["content_lpips"]),
                "method_content_lpips": float(method["content_lpips"]),
                "gen_image": method["gen_image"],
            }
        )

    out_csv = OUT_DIR / f"samst_{label}_idt_aligned_rows.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aligned_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aligned_rows)

    transfer = [r for r in aligned_rows if r["scope"] == "transfer"]
    full_delta = mean(float(r["delta_clip_style"]) for r in aligned_rows)
    transfer_delta = mean(float(r["delta_clip_style"]) for r in transfer)

    return {
        "aligned_csv": str(out_csv.relative_to(REPO)),
        "idt_rows": len(idt_rows),
        "method_rows": len(method_rows),
        "common_rows": len(common),
        "missing_in_method": len(idt_keys - method_keys),
        "missing_in_idt": len(method_keys - idt_keys),
        "transfer_rows": len(transfer),
        "full_delta_clip_style_mean": full_delta,
        "transfer_delta_clip_style_mean": transfer_delta,
        "row_alignment_closed": len(idt_keys - method_keys) == 0 and len(method_keys - idt_keys) == 0,
    }


def _summarize_samst(label: str, paths: dict[str, Path], idt_rows: list[dict[str, str]], compare: dict[str, object]) -> dict[str, object]:
    method_rows = _load_csv(paths["metrics"])
    artfid = json.loads(paths["artfid"].read_text(encoding="utf-8"))
    alignment = _align_rows(label, idt_rows, method_rows)
    compare_row = compare[label]

    full_rows = _scope(method_rows, transfer_only=False)
    transfer_rows = _scope(method_rows, transfer_only=True)
    return {
        "metrics_csv": str(paths["metrics"].relative_to(REPO)),
        "summary_json": str(paths["summary"].relative_to(REPO)),
        "artfid_json": str(paths["artfid"].relative_to(REPO)),
        "full": {
            "count": len(full_rows),
            "clip_style": _avg(full_rows, "clip_style"),
            "content_lpips": _avg(full_rows, "content_lpips"),
            "targetwise_artfid": artfid["full"]["aggregate_art_fid"],
        },
        "transfer": {
            "count": len(transfer_rows),
            "clip_style": _avg(transfer_rows, "clip_style"),
            "content_lpips": _avg(transfer_rows, "content_lpips"),
            "targetwise_artfid": artfid["transfer"]["aggregate_art_fid"],
        },
        "timing": {
            "train_wall_seconds": compare_row.get("train_wall_seconds"),
            "targetwise_artfid_wall_seconds": artfid.get("wall_seconds"),
            "inference_ms_per_img": None,
            "inference_timing_status": "missing_from_current_packet",
        },
        "alignment": alignment,
        "packet_closed": False,
        "missing_for_closed_packet": ["same-scope inference ms/img bound"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idt_rows = _load_csv(IDT_METRICS)
    compare = json.loads(SAMST_COMPARE.read_text(encoding="utf-8"))

    samst = {label: _summarize_samst(label, paths, idt_rows, compare) for label, paths in SAMST.items()}

    status = {
        "date": "2026-06-04",
        "scope": "Distinct5-512 baseline packet status from existing artifacts",
        "idt_metrics": str(IDT_METRICS.relative_to(REPO)),
        "samst": samst,
        "samam": {
            "expected_remote_curve_root": str(SAMAM_EXPECTED_REMOTE),
            "expected_remote_curve_visible_from_this_workspace": SAMAM_EXPECTED_REMOTE.exists(),
            "packet_closed": False,
            "missing_for_closed_packet": [
                "visible final/tuned Distinct5 metrics packet",
                "targetwise ArtFID for final/tuned point",
                "same-scope timing",
                "IDT-aligned per-image rows or explicit missing-row report",
            ],
        },
        "paper_gate": {
            "saMST": "partial: metrics, targetwise ArtFID, train timing, and IDT row alignment are available; inference ms/img is not bound into this packet",
            "saMAM": "open: authoritative Distinct5 curve root is not visible from this workspace and no closed packet is available",
        },
    }

    (OUT_DIR / "packet_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    lines = [
        "# Distinct5-512 baseline packet status",
        "",
        "Date: 2026-06-04",
        "",
        "This packet audits existing artifacts only. It does not rerun training or evaluation.",
        "",
        "## Verdict",
        "",
        "- SaMST is partially closed: full/transfer metrics, targetwise ArtFID, train wall time, and IDT row alignment are available for e5/e15.",
        "- SaMST is not fully closed because same-scope inference `ms/img` is not bound into this packet.",
        "- SaMAM remains open: the authoritative Distinct5 curve root is not visible from this workspace and no final/tuned IDT-aligned packet is available here.",
        "",
        "## SaMST summary",
        "",
        "| checkpoint | tr CLIP-S | tr LPIPS | tr ArtFID | full CLIP-S | full LPIPS | full ArtFID | train h | aligned rows | missing rows | closed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label in ["e5", "e15"]:
        row = samst[label]
        align = row["alignment"]
        train_h = float(row["timing"]["train_wall_seconds"]) / 3600.0
        lines.append(
            f"| {label} | {row['transfer']['clip_style']:.6f} | {row['transfer']['content_lpips']:.6f} | "
            f"{row['transfer']['targetwise_artfid']:.1f} | {row['full']['clip_style']:.6f} | "
            f"{row['full']['content_lpips']:.6f} | {row['full']['targetwise_artfid']:.1f} | "
            f"{train_h:.2f} | {align['common_rows']} | {align['missing_in_method'] + align['missing_in_idt']} | partial |"
        )

    lines += [
        "",
        "Generated aligned-row reports:",
        "",
    ]
    for label in ["e5", "e15"]:
        lines.append(f"- `{samst[label]['alignment']['aligned_csv']}`")

    lines += [
        "",
        "## Missing evidence",
        "",
        "SaMST still needs a same-scope inference `ms/img` timing artifact before it should be called a fully closed packet.",
        "",
        "SaMAM still needs:",
        "",
        "- visible final/tuned Distinct5 metrics packet;",
        "- targetwise ArtFID for the final/tuned point;",
        "- same-scope timing;",
        "- IDT-aligned per-image rows or an explicit missing-row report.",
        "",
        "## Paper implication",
        "",
        "The current paper wording should remain unchanged: SaMST can be reported at e15 with high LPIPS and high targetwise ArtFID, while SaMAM must remain a point-estimate / pending-packet claim.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
