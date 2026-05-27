#!/usr/bin/env python
"""Image-space diagnostics against the Seedream 4.5 golden baseline.

The goal is not to replace CLIP/LPIPS evaluation.  This script measures where a
candidate stylizer differs from the source image and whether those differences
look more boundary-respecting or more like broad color/texture flooding.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EPS = 1e-8


@dataclass(frozen=True)
class MethodRoot:
    name: str
    image_dir: Path


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    root = repo.parent
    default_out = repo / "exp" / "diagnostics" / "seedream_gap"
    return argparse.ArgumentParser(description=__doc__).parse_args(
        None
    ) if False else _parse_args(root, repo, default_out)


def _parse_args(root: Path, repo: Path, default_out: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root
        / "Related_Works"
        / "baseline_pipeline"
        / "results"
        / "seedream45_api"
        / "protocol_a_800"
        / "merged_manifest.csv",
    )
    parser.add_argument(
        "--seedream-dir",
        type=Path,
        default=root
        / "Related_Works"
        / "baseline_pipeline"
        / "results"
        / "seedream45_api"
        / "protocol_a_800"
        / "images",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="name=path to an image directory. Can be repeated.",
    )
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--sheet-top-k", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--focus-target",
        default="",
        help="Optional target style to emphasize in contact sheets and readout, e.g. Hayao.",
    )
    return parser.parse_args()


def default_methods(repo: Path) -> list[MethodRoot]:
    return [
        MethodRoot(
            "t01_original_vae_e8",
            repo
            / "exp"
            / "diffeomorphic_tangent_sweep"
            / "t01_ws0p03_g6_nl0p05"
            / "full_eval"
            / "epoch_0008"
            / "images",
        ),
        MethodRoot(
            "ema_support_w30_guard_e6",
            repo
            / "exp"
            / "diagnostics"
            / "seedream_gap"
            / "inputs"
            / "ema_sconv_support_w30_guard_e6",
        ),
        MethodRoot(
            "ema_support_w40_style_e6",
            repo
            / "exp"
            / "diagnostics"
            / "seedream_gap"
            / "inputs"
            / "ema_sconv_support_w40_style_e6",
        ),
    ]


def parse_method_specs(specs: Iterable[str], repo: Path) -> list[MethodRoot]:
    if not specs:
        return default_methods(repo)
    methods: list[MethodRoot] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--method must be name=path, got: {spec}")
        name, raw_path = spec.split("=", 1)
        methods.append(MethodRoot(name.strip(), Path(raw_path).expanduser()))
    return methods


def read_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("status") and row["status"] != "ok":
                continue
            rows.append(row)
    return rows


def load_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def luminance(x: np.ndarray) -> np.ndarray:
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    p = np.pad(gray, 1, mode="reflect")
    gx = (
        -p[:-2, :-2]
        - 2 * p[1:-1, :-2]
        - p[2:, :-2]
        + p[:-2, 2:]
        + 2 * p[1:-1, 2:]
        + p[2:, 2:]
    )
    gy = (
        -p[:-2, :-2]
        - 2 * p[:-2, 1:-1]
        - p[:-2, 2:]
        + p[2:, :-2]
        + 2 * p[2:, 1:-1]
        + p[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def laplacian(gray: np.ndarray) -> np.ndarray:
    p = np.pad(gray, 1, mode="reflect")
    return 4 * p[1:-1, 1:-1] - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:]


def dilate(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            out |= shifted
    return out


def block_std(gray: np.ndarray, block: int = 16) -> float:
    h, w = gray.shape
    h2 = h - (h % block)
    w2 = w - (w % block)
    if h2 == 0 or w2 == 0:
        return float(np.std(gray))
    blocks = gray[:h2, :w2].reshape(h2 // block, block, w2 // block, block)
    return float(blocks.std(axis=(1, 3)).mean())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(av) * np.linalg.norm(bv) + EPS
    return float(np.dot(av, bv) / denom)


def image_metrics(src: np.ndarray, out: np.ndarray) -> dict[str, float]:
    src_y = luminance(src)
    out_y = luminance(out)
    src_grad = sobel_mag(src_y)
    out_grad = sobel_mag(out_y)
    residual = np.abs(out - src).mean(axis=2)
    edge_thr = np.quantile(src_grad, 0.85)
    flat_thr = np.quantile(src_grad, 0.35)
    edge_mask = dilate(src_grad >= edge_thr, radius=2)
    flat_mask = src_grad <= flat_thr
    out_low_grad = out_grad <= np.quantile(out_grad, 0.45)

    src_hp = laplacian(src_y)
    delta_hp = laplacian(out_y - src_y)
    mean_delta = out.mean(axis=(0, 1)) - src.mean(axis=(0, 1))
    std_delta = out.std(axis=(0, 1)) - src.std(axis=(0, 1))

    flat_residual = float(residual[flat_mask].mean()) if flat_mask.any() else 0.0
    edge_residual = float(residual[edge_mask].mean()) if edge_mask.any() else 0.0
    flat_flood_mask = flat_mask & out_low_grad
    flat_flood = float(residual[flat_flood_mask].mean()) if flat_flood_mask.any() else 0.0

    return {
        "mean_abs_delta": float(residual.mean()),
        "rgb_mean_shift": float(np.linalg.norm(mean_delta)),
        "rgb_std_shift": float(np.linalg.norm(std_delta)),
        "edge_residual": edge_residual,
        "flat_residual": flat_residual,
        "edge_to_flat_ratio": edge_residual / (flat_residual + EPS),
        "flat_color_flood": flat_flood,
        "highpass_delta_energy": float(np.mean(np.abs(delta_hp))),
        "highpass_phase_cos": cosine(src_hp, delta_hp),
        "output_grad_mean": float(out_grad.mean()),
        "output_block_std16": block_std(out_y, block=16),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_keys = [
        key
        for key in rows[0].keys()
        if key
        not in {
            "filename",
            "method",
            "source_style",
            "source_stem",
            "target_style",
            "source_path",
            "output_path",
        }
    ]
    out: list[dict[str, object]] = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["target_style"])), []).append(row)
        groups.setdefault((str(row["method"]), "ALL"), []).append(row)
    for (method, target), items in sorted(groups.items()):
        agg: dict[str, object] = {"method": method, "target_style": target, "count": len(items)}
        for key in metric_keys:
            vals = [float(item[key]) for item in items if item.get(key) not in ("", None)]
            agg[key] = mean(vals)
        out.append(agg)
    return out


def collect_rows(args: argparse.Namespace, methods: list[MethodRoot]) -> list[dict[str, object]]:
    manifest_rows = read_manifest(args.manifest)
    if args.max_images > 0:
        manifest_rows = manifest_rows[: args.max_images]

    seedream = MethodRoot("seedream45_golden", args.seedream_dir)
    all_methods = [seedream] + methods
    metric_rows: list[dict[str, object]] = []
    for row in manifest_rows:
        filename = f"{row['source_style']}_{row['source_stem']}_to_{row['target_style']}.jpg"
        source_path = Path(row["source_path"])
        if not source_path.exists():
            continue
        src = load_rgb(source_path, args.image_size)
        for method in all_methods:
            output_path = method.image_dir / filename
            if not output_path.exists():
                continue
            out_img = load_rgb(output_path, args.image_size)
            values: dict[str, object] = {
                "filename": filename,
                "method": method.name,
                "source_style": row["source_style"],
                "source_stem": row["source_stem"],
                "target_style": row["target_style"],
                "source_path": str(source_path),
                "output_path": str(output_path),
            }
            values.update(image_metrics(src, out_img))
            metric_rows.append(values)
    return metric_rows


def add_golden_gaps(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_file_method = {(str(r["filename"]), str(r["method"])): r for r in rows}
    metric_keys = [
        k
        for k, v in rows[0].items()
        if isinstance(v, float) and not k.endswith("_vs_seedream")
    ]
    out: list[dict[str, object]] = []
    for row in rows:
        new_row = dict(row)
        golden = by_file_method.get((str(row["filename"]), "seedream45_golden"))
        if golden and row["method"] != "seedream45_golden":
            for key in metric_keys:
                new_row[f"{key}_vs_seedream"] = float(row[key]) - float(golden[key])
        else:
            for key in metric_keys:
                new_row[f"{key}_vs_seedream"] = 0.0
        out.append(new_row)
    return out


def add_diagnostic_scores(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Add Seedream-relative failure scores.

    Positive values mean the candidate is worse than Seedream on that axis.
    These are diagnostic proxies, not training targets.
    """
    out: list[dict[str, object]] = []
    for row in rows:
        new_row = dict(row)
        if row["method"] == "seedream45_golden":
            texture = 0.0
            palette = 0.0
            edge = 0.0
            flatness = 0.0
        else:
            texture = max(0.0, float(row.get("highpass_delta_energy_vs_seedream", 0.0))) + max(
                0.0,
                float(row.get("output_block_std16_vs_seedream", 0.0)),
            )
            palette = abs(float(row.get("rgb_mean_shift_vs_seedream", 0.0))) + 0.5 * abs(
                float(row.get("rgb_std_shift_vs_seedream", 0.0))
            )
            edge = max(0.0, -float(row.get("highpass_phase_cos_vs_seedream", 0.0)))
            flatness = max(0.0, float(row.get("output_block_std16_vs_seedream", 0.0))) + max(
                0.0,
                -float(row.get("flat_color_flood_vs_seedream", 0.0)),
            )
        target = str(row.get("target_style", ""))
        hayao_weight = 1.35 if target.lower() == "hayao" else 1.0
        new_row["texture_fragmentation_gap"] = texture
        new_row["palette_shift_gap"] = palette
        new_row["edge_alignment_deficit"] = edge
        new_row["flatness_deficit"] = flatness
        new_row["hayao_failure_score"] = hayao_weight * (texture + 0.55 * palette + 0.45 * edge + 0.35 * flatness)
        out.append(new_row)
    return out


def write_readout(path: Path, summary_rows: list[dict[str, object]], focus_target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = focus_target.strip() or "Hayao"
    target_rows = [
        r
        for r in summary_rows
        if str(r.get("target_style", "")).lower() == target.lower()
        and str(r.get("method", "")) != "seedream45_golden"
    ]
    all_rows = [
        r
        for r in summary_rows
        if str(r.get("target_style", "")) == "ALL"
        and str(r.get("method", "")) != "seedream45_golden"
    ]

    def _fmt_rows(rows: list[dict[str, object]]) -> list[str]:
        rows = sorted(rows, key=lambda r: float(r.get("hayao_failure_score", 0.0)), reverse=True)
        lines = []
        for r in rows:
            lines.append(
                "| {method} | {score:.5f} | {texture:.5f} | {flat:.5f} | {edge:.5f} | {palette:.5f} |".format(
                    method=str(r.get("method", "")),
                    score=float(r.get("hayao_failure_score", 0.0)),
                    texture=float(r.get("texture_fragmentation_gap", 0.0)),
                    flat=float(r.get("flatness_deficit", 0.0)),
                    edge=float(r.get("edge_alignment_deficit", 0.0)),
                    palette=float(r.get("palette_shift_gap", 0.0)),
                )
            )
        return lines

    lines = [
        "# Seedream Gap Diagnostic Readout",
        "",
        "Seedream 4.5 is used here as a diagnostic visual reference only. These scores are not training losses.",
        "",
        f"## Focus Target: {target}",
        "",
        "| method | failure | texture | flatness | edge | palette |",
        "|---|---:|---:|---:|---:|---:|",
        *_fmt_rows(target_rows),
        "",
        "## All Targets",
        "",
        "| method | failure | texture | flatness | edge | palette |",
        "|---|---:|---:|---:|---:|---:|",
        *_fmt_rows(all_rows),
        "",
        "Interpretation:",
        "",
        "- high texture means fragmented high-frequency edits relative to Seedream;",
        "- high flatness means the output did not form clean color planes;",
        "- high edge means the edit is less phase-aligned with source boundaries;",
        "- high palette means broad color statistics drift differently from the reference.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_contact_sheet(rows: list[dict[str, object]], methods: list[MethodRoot], args: argparse.Namespace) -> Path:
    by_file_method = {(str(r["filename"]), str(r["method"])): r for r in rows}
    candidates = [
        r
        for r in rows
        if r["method"] != "seedream45_golden"
        and "flat_color_flood_vs_seedream" in r
        and str(r["target_style"]) != "photo"
    ]
    focus_target = str(getattr(args, "focus_target", "") or "").strip()
    if focus_target:
        focused = [r for r in candidates if str(r["target_style"]).lower() == focus_target.lower()]
        if focused:
            candidates = focused
    candidates.sort(
        key=lambda r: (
            float(r.get("hayao_failure_score", 0.0))
            + 0.25 * float(r["flat_color_flood_vs_seedream"])
        ),
        reverse=True,
    )

    selected_files: list[str] = []
    for row in candidates:
        filename = str(row["filename"])
        if filename not in selected_files:
            selected_files.append(filename)
        if len(selected_files) >= args.sheet_top_k:
            break

    cols = ["source", "seedream45"] + [m.name for m in methods]
    thumb = 160
    label_h = 28
    pad = 8
    width = len(cols) * (thumb + pad) + pad
    height = len(selected_files) * (thumb + label_h + pad) + pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for r_idx, filename in enumerate(selected_files):
        golden = by_file_method[(filename, "seedream45_golden")]
        source_path = Path(str(golden["source_path"]))
        image_paths = [source_path, args.seedream_dir / filename] + [m.image_dir / filename for m in methods]
        y = pad + r_idx * (thumb + label_h + pad)
        for c_idx, (label, path) in enumerate(zip(cols, image_paths)):
            x = pad + c_idx * (thumb + pad)
            draw.text((x, y), label[:24], fill=(0, 0, 0), font=font)
            if path.exists():
                with Image.open(path) as im:
                    im = im.convert("RGB").resize((thumb, thumb), Image.Resampling.BICUBIC)
                sheet.paste(im, (x, y + label_h))
            else:
                draw.rectangle((x, y + label_h, x + thumb, y + label_h + thumb), outline=(200, 0, 0))
                draw.text((x + 4, y + label_h + 4), "missing", fill=(200, 0, 0), font=font)

    path = args.out_dir / "seedream_gap_worst_cases.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    methods = parse_method_specs(args.method, repo)
    rows = collect_rows(args, methods)
    if not rows:
        raise SystemExit("No comparable rows found.")
    rows = add_golden_gaps(rows)
    rows = add_diagnostic_scores(rows)
    summary_rows = aggregate(rows)
    write_csv(args.out_dir / "seedream_gap_image_metrics.csv", rows)
    write_csv(args.out_dir / "seedream_gap_summary.csv", summary_rows)
    write_readout(args.out_dir / "seedream_gap_readout.md", summary_rows, args.focus_target)
    sheet = make_contact_sheet(rows, methods, args)
    print(f"rows={len(rows)}")
    print(f"summary={args.out_dir / 'seedream_gap_summary.csv'}")
    print(f"details={args.out_dir / 'seedream_gap_image_metrics.csv'}")
    print(f"readout={args.out_dir / 'seedream_gap_readout.md'}")
    print(f"sheet={sheet}")


if __name__ == "__main__":
    main()
