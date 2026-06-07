from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
DOC_ROOT = ROOT / "docs" / "experiments"
IDT_TRANSFER_CLIP = 0.6399208252628644

INMORTAL_EPOCH_CSV = DOC_ROOT / "2026-06-07-inmortal-epoch-eval-table.csv"
INMORTAL_MASTER_CSV = DOC_ROOT / "aaai2027_inmortal_results_master.csv"
RESULTS_MASTER_CSV = DOC_ROOT / "aaai2027_results_master.csv"
OUT_CSV = WORKSPACE / "best.csv"


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _metric_row(rows: list[dict[str, str]], *, clip_key: str, lpips_key: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        clip_style = _safe_float(row.get(clip_key))
        content_lpips = _safe_float(row.get(lpips_key))
        if clip_style is None or content_lpips is None:
            continue
        item = dict(row)
        item["clip_style"] = clip_style
        item["content_lpips"] = content_lpips
        out.append(item)
    return out


def _pick_max_style(rows: list[dict[str, object]]) -> dict[str, object]:
    return max(rows, key=lambda row: (float(row["clip_style"]), -float(row["content_lpips"])))


def _pick_min_lpips(rows: list[dict[str, object]], *, min_style: float) -> dict[str, object] | None:
    keep = [row for row in rows if float(row["clip_style"]) >= min_style]
    if not keep:
        return None
    return min(keep, key=lambda row: (float(row["content_lpips"]), -float(row["clip_style"])))


def _epoch_entry(
    *,
    rank: int,
    slot: str,
    row: dict[str, object],
    note: str,
) -> dict[str, object]:
    run_name = str(row["run_name"])
    epoch = str(row["epoch"])
    return {
        "rank": rank,
        "slot": slot,
        "experiment": run_name,
        "clip_style": row["clip_style"],
        "content_lpips": row["content_lpips"],
        "one_minus_lpips": 1.0 - float(row["content_lpips"]),
        "delta_idt_tr": float(row["clip_style"]) - IDT_TRANSFER_CLIP,
        "train_time_sec": _safe_float(row.get("train_time_sec")),
        "train_time_min": (_safe_float(row.get("train_time_sec")) or 0.0) / 60.0 if _safe_float(row.get("train_time_sec")) is not None else None,
        "full_clip_style": _safe_float(row.get("full_clip_style")),
        "full_content_lpips": _safe_float(row.get("full_content_lpips")),
        "selection": epoch,
        "source_table": str(INMORTAL_EPOCH_CSV.relative_to(WORKSPACE)),
        "evidence_path": row.get("summary_path", ""),
        "note": note,
    }


def _final_entry(
    *,
    rank: int,
    slot: str,
    row: dict[str, object],
    note: str,
    source_table: Path,
) -> dict[str, object]:
    train_time_min = _safe_float(row.get("train_wall"))
    return {
        "rank": rank,
        "slot": slot,
        "experiment": row["experiment"],
        "clip_style": row["clip_style"],
        "content_lpips": row["content_lpips"],
        "one_minus_lpips": 1.0 - float(row["content_lpips"]),
        "delta_idt_tr": float(row["clip_style"]) - IDT_TRANSFER_CLIP,
        "train_time_sec": train_time_min * 60.0 if train_time_min is not None else None,
        "train_time_min": train_time_min,
        "full_clip_style": _safe_float(row.get("full_clip_style")),
        "full_content_lpips": _safe_float(row.get("full_content_lpips")),
        "selection": row.get("selection", ""),
        "source_table": str(source_table.relative_to(WORKSPACE)),
        "evidence_path": row.get("evidence_path", ""),
        "note": note,
    }


def main() -> int:
    epoch_rows = _metric_row(_read_csv(INMORTAL_EPOCH_CSV), clip_key="clip_style", lpips_key="content_lpips")
    inmortal_final = _metric_row(_read_csv(INMORTAL_MASTER_CSV), clip_key="transfer_clip_style", lpips_key="transfer_content_lpips")
    results_master = _metric_row(_read_csv(RESULTS_MASTER_CSV), clip_key="clip_style", lpips_key="content_lpips")

    compact_rows = [
        row
        for row in results_master
        if row.get("dataset") == "distinct5_512"
        and row.get("method") == "LBM"
        and str(row.get("variant")) in {"F_e1", "H_e1", "H_e2", "K_e1"}
    ]
    kinetic_rows = [row for row in epoch_rows if str(row.get("family", "")).startswith("K_")]
    structot_rows = [row for row in epoch_rows if "StructOT" in str(row.get("family", ""))]
    structot_final_rows = [row for row in inmortal_final if row.get("experiment") == "inmortal_xpred_structot_seed42_b16"]
    kinetic_final_rows = [row for row in inmortal_final if row.get("experiment") == "inmortal_k_manifold_seed42_b16"]

    style_best = _pick_max_style(epoch_rows)
    balanced_candidates = [
        row
        for row in epoch_rows
        if "Pattn_Stokes" in str(row.get("family", ""))
        and float(row["clip_style"]) >= 0.72
    ]
    balanced_best = min(
        balanced_candidates,
        key=lambda row: (float(row["content_lpips"]), -float(row["clip_style"])),
    ) if balanced_candidates else None
    promoted_lpips70_candidates = [
        row
        for row in inmortal_final
        if float(row["clip_style"]) >= 0.70
    ]
    promoted_lpips70_best = min(
        promoted_lpips70_candidates,
        key=lambda row: (float(row["content_lpips"]), -float(row["clip_style"])),
    ) if promoted_lpips70_candidates else None
    structot_best = structot_final_rows[0] if structot_final_rows else (_pick_min_lpips(structot_rows, min_style=0.70) if structot_rows else None)
    kinetic_best = kinetic_final_rows[0] if kinetic_final_rows else (_pick_max_style(kinetic_rows) if kinetic_rows else None)
    compact_style = _pick_max_style(compact_rows) if compact_rows else None

    rows: list[dict[str, object]] = []
    rows.append(
        _epoch_entry(
            rank=1,
            slot="style_best_current",
            row=style_best,
            note="Highest current Distinct5-512 transfer CLIP-style among tracked inmortal epochs.",
        )
    )
    if balanced_best is not None:
        rows.append(
            _epoch_entry(
                rank=2,
                slot="balanced_best_current",
                row=balanced_best,
                note="Best current frontier balance among the promoted Pattn/Stokes lines.",
            )
        )
    if promoted_lpips70_best is not None:
        rows.append(
            _final_entry(
                rank=3,
                slot="best_promoted_lpips_ge_070",
                row=promoted_lpips70_best,
                note="Lowest-LPIPS paper-facing promoted point with transfer CLIP-style >= 0.70.",
                source_table=INMORTAL_MASTER_CSV,
            )
        )
    if structot_best is not None:
        rows.append(
            (_final_entry if "experiment" in structot_best else _epoch_entry)(
                rank=4,
                slot="best_structot_tradeoff",
                row=structot_best,
                note="Best current StructOT tradeoff point and strongest lower-LPIPS secondary line.",
                source_table=INMORTAL_MASTER_CSV,
            )
        )
    if kinetic_best is not None:
        rows.append(
            (_final_entry if "experiment" in kinetic_best else _epoch_entry)(
                rank=5,
                slot="best_kinetic_only_control",
                row=kinetic_best,
                note="Best pure kinetic control on the tracked Distinct5-512 epoch surface.",
                source_table=INMORTAL_MASTER_CSV,
            )
        )
    if compact_style is not None:
        rows.append(
            _final_entry(
                rank=6,
                slot="best_compact_mainline_anchor",
                row=compact_style,
                note="Best compact earlier-mainline LBM anchor for paper-facing comparison.",
                source_table=RESULTS_MASTER_CSV,
            )
        )

    fieldnames = [
        "rank",
        "clip_style",
        "content_lpips",
        "one_minus_lpips",
        "delta_idt_tr",
        "train_time_sec",
        "train_time_min",
        "experiment",
        "slot",
        "selection",
        "full_clip_style",
        "full_content_lpips",
        "evidence_path",
        "note",
        "source_table",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(OUT_CSV)
    for row in rows:
        print(f"{row['slot']}: {row['experiment']} {row['selection']} {row['clip_style']:.6f} / {row['content_lpips']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
