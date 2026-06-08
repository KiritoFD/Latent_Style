from __future__ import annotations

import csv
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "final"
FINAL.mkdir(parents=True, exist_ok=True)

FILES_TO_COPY = [
    ROOT / "paper_aaai2027.tex",
    ROOT / "paper_aaai2027.pdf",
    ROOT / "supplement_aaai2027.tex",
    ROOT / "supplement_aaai2027.pdf",
    ROOT / "ACTIVE_DRAFT.md",
    ROOT / "official_requirement_check.md",
    ROOT / "review_closure_status.md",
    ROOT / "strong_accept_gap_note.md",
    ROOT / "main_point_artifact_ledger.csv",
    ROOT / "distinct5_aux_artifact_table.csv",
    ROOT / "distinct5_nonclip_style_probe.csv",
    ROOT / "distinct5_idt_bootstrap_extended.csv",
    ROOT / "paper_point_param_counts.csv",
    ROOT / "local_analysis_requirements.txt",
    ROOT / "figures" / "fig_distinct5_page1_summary.pdf",
    ROOT / "figures" / "fig_distinct5_qualitative_main.png",
    ROOT / "figures" / "fig_distinct5_qualitative_appendix_a.png",
    ROOT / "figures" / "fig_distinct5_qualitative_appendix_b.png",
    ROOT / "framework_lbm_main_user.png",
    ROOT / "blind_pairwise_v1" / "README.md",
    ROOT / "blind_pairwise_v1" / "exploratory_blind_audit.csv",
    ROOT / "blind_pairwise_v1" / "exploratory_blind_audit.md",
    ROOT / "blind_pairwise_v1" / "exploratory_blind_audit_summary.csv",
]


def relabel_destination(src: Path) -> Path:
    mapping = {
        ROOT / "blind_pairwise_v1" / "README.md": FINAL / "blind_pairwise_README.md",
        ROOT / "blind_pairwise_v1" / "exploratory_blind_audit.csv": FINAL / "blind_pairwise_exploratory_blind_audit.csv",
        ROOT / "blind_pairwise_v1" / "exploratory_blind_audit.md": FINAL / "blind_pairwise_exploratory_blind_audit.md",
        ROOT / "blind_pairwise_v1" / "exploratory_blind_audit_summary.csv": FINAL / "blind_pairwise_exploratory_blind_audit_summary.csv",
    }
    return mapping.get(src, FINAL / src.name)


def main() -> None:
    copied_rows = []
    for src in FILES_TO_COPY:
        if not src.exists():
            raise FileNotFoundError(src)
        dst = relabel_destination(src)
        shutil.copy2(src, dst)
        copied_rows.append(
            {
                "source": str(src),
                "destination": str(dst),
                "bytes": dst.stat().st_size,
            }
        )

    manifest = FINAL / "aaai27_submission_bundle_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(copied_rows[0].keys()))
        writer.writeheader()
        writer.writerows(copied_rows)

    bundle = FINAL / "aaai27_submission_bundle_current.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for row in copied_rows:
            dst = Path(row["destination"])
            zf.write(dst, arcname=dst.name)
        zf.write(manifest, arcname=manifest.name)

    print(bundle)
    print(manifest)


if __name__ == "__main__":
    main()
