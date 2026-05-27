from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "exp"
MANIFEST_DIR = EXP / "_cleanup_manifests"


EXTRA_KEEP_EVALS = {
    # Paper-visible anchors.
    "diffeomorphic_tangent_sweep/t01_ws0p03_g6_nl0p05/full_eval/epoch_0008",
    "diffeomorphic_tangent_sweep/t00_ws0p03_g6_nl0/full_eval/epoch_0008",
    "frontier_decision_tree_8h/31_t01_pair8_bridge_tax/full_eval/epoch_0007",
    "frontier_decision_tree_8h/15_t01_pair8_led9/full_eval/epoch_0008",
    "scale/wikiart_quarter_bs4/full_eval/epoch_0001",
    # DINO / pairing conclusion points.
    "orthogonal_budget36/26_dino_t00_zero_top8/full_eval/epoch_0007",
    "orthogonal_budget36/27_dino_t00_zero_top4/full_eval/epoch_0008",
    # Patch and stagewise probes worth keeping as endpoints.
    "stagewise_meeting/00_p8tax_relax/full_eval/epoch_0010",
}

EXTRA_KEEP_CKPTS = {
    "diffeomorphic_tangent_sweep/t01_ws0p03_g6_nl0p05/epoch_0008.pt",
    "diffeomorphic_tangent_sweep/t00_ws0p03_g6_nl0/epoch_0008.pt",
    "frontier_decision_tree_8h/31_t01_pair8_bridge_tax/epoch_0007.pt",
    "frontier_decision_tree_8h/15_t01_pair8_led9/epoch_0008.pt",
    "scale/wikiart_quarter_bs4/epoch_0001.pt",
    "orthogonal_budget36/26_dino_t00_zero_top8/epoch_0007.pt",
    "orthogonal_budget36/27_dino_t00_zero_top4/epoch_0008.pt",
    "stagewise_meeting/00_p8tax_relax/epoch_0010.pt",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(EXP.resolve()).as_posix()


def parse_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    overview = ((payload.get("analysis") or {}).get("all_pairs_overview") or {})
    style = overview.get("clip_style")
    lpips = overview.get("content_lpips")
    ec = None
    if isinstance(style, (int, float)) and isinstance(lpips, (int, float)):
        ec = float(style) * (1.0 - float(lpips))
    parts = path.relative_to(EXP).parts
    run = "/".join(parts[:-1])
    epoch = ""
    if "full_eval" in parts:
        idx = parts.index("full_eval")
        run = "/".join(parts[:idx])
        if idx + 1 < len(parts):
            epoch = parts[idx + 1]
    return {
        "summary_path": rel(path),
        "exp_group": parts[0] if parts else "",
        "run": run,
        "epoch": epoch,
        "checkpoint": payload.get("checkpoint", ""),
        "clip_style": style,
        "clip_content": overview.get("clip_content"),
        "content_lpips": lpips,
        "clip_dir": overview.get("clip_dir"),
        "ec": ec,
    }


def ckpt_rel_from_summary_value(value: str) -> str | None:
    if not value or "reuse-only" in value:
        return None
    normalized = value.replace("\\", "/")
    marker = "/SchrodingerBridge/exp/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    marker = "SchrodingerBridge/exp/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    marker = "exp/"
    if normalized.startswith(marker):
        return normalized[len(marker) :]
    path = Path(normalized)
    try:
        return path.resolve().relative_to(EXP.resolve()).as_posix()
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_keep_sets() -> tuple[set[str], set[str], list[dict[str, Any]]]:
    summaries = [parse_summary(p) for p in EXP.rglob("summary.json")]
    scored = [r for r in summaries if isinstance(r.get("ec"), float)]

    keep_evals = set(EXTRA_KEEP_EVALS)
    keep_ckpts = set(EXTRA_KEEP_CKPTS)
    representatives: list[dict[str, Any]] = []

    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        by_group.setdefault(str(row["exp_group"]), []).append(row)

    for group, rows in by_group.items():
        best = max(rows, key=lambda r: float(r["ec"]))
        representatives.append({"reason": "best_ec_per_group", **best})
        if "/full_eval/" in best["summary_path"]:
            keep_evals.add(best["summary_path"].rsplit("/summary.json", 1)[0])
        ckpt = ckpt_rel_from_summary_value(str(best.get("checkpoint", "")))
        if ckpt:
            keep_ckpts.add(ckpt)

    for eval_rel in sorted(EXTRA_KEEP_EVALS):
        summary = EXP / eval_rel / "summary.json"
        row = parse_summary(summary) if summary.exists() else {
            "summary_path": f"{eval_rel}/summary.json",
            "exp_group": eval_rel.split("/", 1)[0],
            "run": eval_rel,
            "epoch": "",
            "checkpoint": "",
            "clip_style": None,
            "clip_content": None,
            "content_lpips": None,
            "clip_dir": None,
            "ec": None,
        }
        representatives.append({"reason": "explicit_keep", **row})

    return keep_evals, keep_ckpts, representatives


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune SchrodingerBridge/exp heavy artifacts while keeping configs/src/results.")
    parser.add_argument("--execute", action="store_true", help="Actually delete files/directories. Default is dry-run.")
    args = parser.parse_args()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    keep_evals, keep_ckpts, representatives = collect_keep_sets()

    all_ckpts = [
        p for p in EXP.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pt", ".ckpt", ".pth", ".model"}
    ]
    delete_ckpts = [p for p in all_ckpts if rel(p) not in keep_ckpts]

    all_full_evals = [p for p in EXP.rglob("full_eval") if p.is_dir()]
    delete_full_eval_children: list[Path] = []
    for full_eval in all_full_evals:
        for child in full_eval.iterdir():
            if child.is_dir():
                child_rel = rel(child)
                if child_rel not in keep_evals:
                    delete_full_eval_children.append(child)

    ckpt_rows = [{"path": rel(p), "bytes": p.stat().st_size, "kept": rel(p) in keep_ckpts} for p in all_ckpts]
    del_ckpt_rows = [{"path": rel(p), "bytes": p.stat().st_size} for p in delete_ckpts]
    del_eval_rows = []
    for p in delete_full_eval_children:
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        del_eval_rows.append({"path": rel(p), "bytes": size})

    write_csv(MANIFEST_DIR / "representative_points_kept.csv", representatives, [
        "reason", "summary_path", "exp_group", "run", "epoch", "checkpoint",
        "clip_style", "clip_content", "content_lpips", "clip_dir", "ec",
    ])
    write_csv(MANIFEST_DIR / "checkpoint_cleanup_plan.csv", ckpt_rows, ["path", "bytes", "kept"])
    write_csv(MANIFEST_DIR / "checkpoint_delete_manifest.csv", del_ckpt_rows, ["path", "bytes"])
    write_csv(MANIFEST_DIR / "full_eval_delete_manifest.csv", del_eval_rows, ["path", "bytes"])

    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "keep_ckpts": len(keep_ckpts),
        "all_ckpts": len(all_ckpts),
        "delete_ckpts": len(delete_ckpts),
        "delete_ckpt_gb": round(sum(r["bytes"] for r in del_ckpt_rows) / 1024 ** 3, 3),
        "keep_full_eval_points": len(keep_evals),
        "delete_full_eval_points": len(delete_full_eval_children),
        "delete_full_eval_gb": round(sum(r["bytes"] for r in del_eval_rows) / 1024 ** 3, 3),
        "manifest_dir": str(MANIFEST_DIR),
    }, indent=2))

    if not args.execute:
        return 0

    for p in delete_ckpts:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    for p in delete_full_eval_children:
        shutil.rmtree(p, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
