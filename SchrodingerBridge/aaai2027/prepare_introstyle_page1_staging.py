from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
STAGING_ROOT = ROOT / "introstyle_page1" / "staging"

LOCAL_POINTS = [
    {
        "run": "SaMST_e15",
        "images_dir": WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samst_distinct5_512_real_b2_e15_20260602" / "eval_epoch15" / "epoch_0015" / "images",
        "metrics_csv": WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samst_distinct5_512_real_b2_e15_20260602" / "eval_epoch15" / "epoch_0015" / "metrics.csv",
    },
    {
        "run": "LBM-Knee_e13",
        "images_dir": ROOT / "local_eval" / "lbm_knee_e13_artfid" / "images",
        "metrics_csv": ROOT / "local_eval" / "lbm_knee_e13_artfid" / "metrics.csv",
    },
    {
        "run": "LBM-PS-v2_e13",
        "images_dir": ROOT / "local_eval" / "pattn_stokes002_e13" / "images",
        "metrics_csv": ROOT / "local_eval" / "pattn_stokes002_e13" / "metrics.csv",
    },
    {
        "run": "Seedream_repaired750",
        "images_dir": WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "seedream45_api" / "distinct5_512_seedream45_windhub_20260607_repaired750" / "images",
        "metrics_csv": WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "seedream45_api" / "distinct5_512_seedream45_windhub_20260607_repaired750" / "metrics.csv",
    },
]

REMOTE_POINTS = [
    {
        "run": "SaMAM_2250",
        "images_dir": Path("/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag/eval_curve/step_002250/images"),
        "metrics_csv": Path("/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag/eval_curve/step_002250/metrics.csv"),
    },
    {
        "run": "Lat_SaMAM_step1500",
        "images_dir": Path("/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_distinct5_512_convergence_20260607_011328/eval_bundle_fast_step1500/step1500_fast/images"),
        "metrics_csv": Path("/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_distinct5_512_convergence_20260607_011328/eval_bundle_fast_step1500/step1500_fast/metrics.csv"),
    },
    {
        "run": "Lat_SaMST_batch1050",
        "images_dir": Path("/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_distinct5_512_convergence_20260606_214051/eval_bundle_fast/batch1050_fast/images"),
        "metrics_csv": Path("/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_distinct5_512_convergence_20260606_214051/eval_bundle_fast/batch1050_fast/metrics.csv"),
    },
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_image(images_dir: Path, gen_image: str) -> Path:
    name = Path(str(gen_image)).name
    direct = images_dir / name
    if direct.exists():
        return direct
    raw = images_dir / str(gen_image)
    if raw.exists():
        return raw
    raise FileNotFoundError(f"{name} under {images_dir}")


def image_cell(row: dict[str, str]) -> str:
    for key in ("gen_image", "image"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise KeyError("Expected one of gen_image/image in metrics row")


def row_src_stem(row: dict[str, str]) -> str:
    src_style = str(row.get("src_style", "")).strip()
    if "src_image" in row and str(row.get("src_image", "")).strip():
        stem = Path(str(row["src_image"])).stem
    elif "src_stem" in row and str(row.get("src_stem", "")).strip():
        stem = Path(str(row["src_stem"])).stem
    else:
        raise KeyError("Expected one of src_image/src_stem in metrics row")
    prefix = f"{src_style}__"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return stem


def load_case_manifest(path: Path) -> set[tuple[str, str, str]]:
    rows = read_rows(path)
    keep: set[tuple[str, str, str]] = set()
    for row in rows:
        keep.add((str(row["src_style"]).strip(), str(row["tgt_style"]).strip(), str(row["src_stem"]).strip()))
    return keep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rows", type=int, default=20)
    parser.add_argument("--profile", choices=["local_only", "remote_only"], default="local_only")
    parser.add_argument("--case-manifest", type=Path)
    args = parser.parse_args()

    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    points = LOCAL_POINTS if args.profile == "local_only" else REMOTE_POINTS
    keep_cases = load_case_manifest(args.case_manifest) if args.case_manifest else None
    for point in points:
        rows = read_rows(Path(point["metrics_csv"]))
        if keep_cases is None:
            keep = rows[: int(args.sample_rows)]
        else:
            keep = [
                row
                for row in rows
                if (str(row.get("src_style", "")).strip(), str(row.get("tgt_style", "")).strip(), row_src_stem(row)) in keep_cases
            ]
        if not keep:
            continue
        out_dir = STAGING_ROOT / str(point["run"])
        images_out = out_dir / "images"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        images_out.mkdir(parents=True, exist_ok=True)
        for row in keep:
            src = resolve_image(Path(point["images_dir"]), image_cell(row))
            shutil.copy2(src, images_out / src.name)
        write_rows(out_dir / "metrics.csv", keep, list(keep[0].keys()))
        print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
