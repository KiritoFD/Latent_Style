"""Collect strict protocol-750 outputs into one evaluation directory."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
RUN511_ROOT = THIS_DIR.parent
WORKSPACE_ROOT = RUN511_ROOT.parent.parent
REFERENCE = (
    WORKSPACE_ROOT
    / "SchrodingerBridge"
    / "exp"
    / "pareto_probe_4"
    / "S-add__K-3_C-2_W-10_Col-15"
    / "full_eval"
    / "epoch_0001"
    / "images"
)

RUNS = {
    "ours_epoch_0007": WORKSPACE_ROOT
    / "SchrodingerBridge"
    / "S-add__K-1_C-0_W-20_Col-0"
    / "full_eval"
    / "epoch_0007"
    / "images",
    "samst_strict": RUN511_ROOT / "outputs" / "samst_750_strict" / "infer_750" / "images",
    "styleid_strict": RUN511_ROOT / "outputs" / "styleid_750_strict" / "infer_750" / "images",
    "s2wat_strict": RUN511_ROOT / "outputs" / "s2wat_750_strict" / "infer_750" / "images",
    "adain_v32k": RUN511_ROOT / "outputs" / "adain_7g_v32k" / "infer_750" / "images",
    "adain_vgg19": RUN511_ROOT / "outputs" / "adain_7g_vgg19" / "infer_750" / "images",
    "adain_bad": RUN511_ROOT / "outputs" / "adain_4g_real" / "infer_750" / "images",
}


def main() -> int:
    out_root = RUN511_ROOT / "complete_750"
    ref_names = sorted(p.name for p in REFERENCE.glob("*.jpg"))
    rows = []

    for name, src_dir in RUNS.items():
        dst = out_root / name / "images"
        dst.mkdir(parents=True, exist_ok=True)
        for old in dst.glob("*.jpg"):
            old.unlink()

        src_names = {p.name for p in src_dir.glob("*.jpg")} if src_dir.exists() else set()
        copied = 0
        for fname in ref_names:
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, dst / fname)
                copied += 1
        dst_names = {p.name for p in dst.glob("*.jpg")}
        rows.append(
            {
                "run": name,
                "src_dir": str(src_dir),
                "images": len(dst_names),
                "ref_match": len(dst_names & set(ref_names)),
                "missing": len(set(ref_names) - dst_names),
                "extra_in_source": len(src_names - set(ref_names)),
                "status": "ok" if copied == len(ref_names) else "partial",
            }
        )

    manifest = {"reference": str(REFERENCE), "runs": rows}
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out_root)
    for row in rows:
        print(row)
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
