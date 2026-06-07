from __future__ import annotations

import argparse
import csv
import io
from collections import OrderedDict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENTS_DIR = SB_ROOT / "docs" / "experiments"
OUTPUT_CSV = EXPERIMENTS_DIR / "aaai2027_results_master.csv"

FIELDNAMES = [
    "experiment",
    "clip_style",
    "content_lpips",
    "metric_surface",
    "dataset",
    "method",
    "variant",
    "selection",
    "train_batch",
    "train_epochs",
    "full_clip_style",
    "full_content_lpips",
    "delta_idt_transfer",
    "train_wall",
    "status",
    "decision",
    "source_csv",
    "evidence_path",
]

LOG_EXCLUDE_VARIANTS = {
    "F_best_lpips",
    "H_balanced",
    "K_best_style",
    "K_longer_e5",
    "e5",
    "e15",
    "step_2250",
    "step_3000",
    "best_current_curve",
    "style_high_damage",
    "latent_samecost_b50_fast",
    "latent_samecost_b300_fast",
    "inmortal_k_spatial_b16",
    "inmortal_k_manifold_b16",
    "inmortal_xpred_bary_b40",
    "inmortal_xpred_kmanifold_b32",
    "inmortal_xpred_phighpass_b28",
    "inmortal_xpred_kmanifold_phighpass_b32",
    "inmortal_xpred_kmanifold_pmod_b32",
    "inmortal_xpred_kmanifold_pattn_b16",
    "inmortal_xpred_kmanifold_pattn_b16_e12_continue",
    "inmortal_xpred_kmanifold_pattn_stokes_from_pattn_b16",
    "inmortal_xpred_kmanifold_pattn_stokes002_from_pattn_b16",
    "inmortal_xpred_kmanifold_pattn_aniso_b16",
    "inmortal_xpred_kmanifold_pattn_stokes_b16",
    "inmortal_xpred_kmanifold_pattn_stokes_b16_e12_continue",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def numeric_like(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def ordered_row(**kwargs: str) -> OrderedDict[str, str]:
    return OrderedDict((field, kwargs.get(field, "")) for field in FIELDNAMES)


def build_rows() -> list[OrderedDict[str, str]]:
    rows: list[OrderedDict[str, str]] = []
    seen: set[str] = set()

    def add(row: OrderedDict[str, str]) -> None:
        key = row["experiment"]
        if key in seen:
            return
        seen.add(key)
        rows.append(row)

    same_cost = read_csv(EXPERIMENTS_DIR / "2026-06-04-distinct5_same_cost_inventory.csv")
    for item in same_cost:
        add(
            ordered_row(
                experiment=f"{item['dataset']}__{item['method']}__{item['arm']}",
                clip_style=item["transfer_clip_style"],
                content_lpips=item["transfer_lpips"],
                metric_surface="transfer",
                dataset=item["dataset"],
                method=item["method"],
                variant=item["arm"],
                selection=item["arm"],
                full_clip_style=item["full_clip_style"],
                full_content_lpips=item["full_lpips"],
                delta_idt_transfer=item["delta_idt_transfer"],
                train_wall=item["train_wall_min"],
                status=item["status"],
                decision=item["status"],
                source_csv="2026-06-04-distinct5_same_cost_inventory.csv",
                evidence_path=item["authoritative_surface"],
            )
        )

    latent_convergence = read_csv(EXPERIMENTS_DIR / "2026-06-07-distinct5_latent_baseline_convergence.csv")
    for item in latent_convergence:
        add(
            ordered_row(
                experiment=f"distinct5_512__{item['method']}__{item['point_type']}__{item['selection']}",
                clip_style=item["transfer_clip_style"],
                content_lpips=item["transfer_content_lpips"],
                metric_surface="transfer",
                dataset="distinct5_512",
                method=item["method"],
                variant=item["point_type"],
                selection=item["selection"],
                full_clip_style=item["full_clip_style"],
                full_content_lpips=item["full_content_lpips"],
                delta_idt_transfer=item["delta_idt_transfer"],
                status="completed",
                decision="keep",
                source_csv="2026-06-07-distinct5_latent_baseline_convergence.csv",
                evidence_path=item["summary_json"],
            )
        )

    inmortal = read_csv(EXPERIMENTS_DIR / "aaai2027_inmortal_results_master.csv")
    for item in inmortal:
        add(
            ordered_row(
                experiment=item["experiment"],
                clip_style=item["transfer_clip_style"],
                content_lpips=item["transfer_content_lpips"],
                metric_surface="transfer",
                dataset="distinct5_512",
                method="LBM",
                variant=item["family"],
                selection=item["selection"],
                train_batch=item["train_batch"],
                train_epochs=item["train_epochs"],
                full_clip_style=item["full_clip_style"],
                full_content_lpips=item["full_content_lpips"],
                status=item["status"],
                decision=item["status"],
                source_csv="aaai2027_inmortal_results_master.csv",
                evidence_path=item["evidence_path"],
            )
        )

    master_log = read_csv(EXPERIMENTS_DIR / "aaai2027_master_experiment_log.csv")
    for item in master_log:
        if not numeric_like(item.get("clip_style", "")) or not numeric_like(item.get("content_lpips", "")):
            continue
        if item["variant_or_point"] in LOG_EXCLUDE_VARIANTS:
            continue
        metric_surface = "reported"
        if item["family"] in {"reviewer_control", "latent_baseline"}:
            metric_surface = "transfer"
        elif item["family"] == "mainline_improvement" and (
            item["variant_or_point"].startswith("A1")
            or item["variant_or_point"].startswith("A2")
            or item["variant_or_point"].startswith("F_longer")
            or item["variant_or_point"].startswith("K_longer")
        ):
            metric_surface = "transfer"
        add(
            ordered_row(
                experiment=f"{item['dataset']}__{item['method']}__{item['variant_or_point']}",
                clip_style=item["clip_style"],
                content_lpips=item["content_lpips"],
                metric_surface=metric_surface,
                dataset=item["dataset"],
                method=item["method"],
                variant=item["family"],
                selection=item["checkpoint_or_step"],
                delta_idt_transfer=item["delta_idt_transfer"],
                train_wall=item["train_wall"],
                status=item["status"],
                decision=item["keep_decision"],
                source_csv="aaai2027_master_experiment_log.csv",
                evidence_path=item["evidence_path"],
            )
        )

    return sorted(rows, key=lambda row: (row["dataset"], row["method"], row["experiment"]))


def render_csv(rows: list[OrderedDict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the unified AAAI2027 experiment results CSV.")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--check", action="store_true", help="Exit nonzero if the output file is stale.")
    args = parser.parse_args()

    rendered = render_csv(build_rows())
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        if current != rendered:
            print(f"stale: {args.output}")
            return 1
        print(f"up_to_date: {args.output}")
        return 0

    with args.output.open("w", encoding="utf-8", newline="") as f:
        f.write(rendered)
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
