from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
SOURCE_ROOT = WORKSPACE / "Dataset" / "distinct5_512" / "test"
OUT_ROOT = ROOT / "blind_pairwise_v1"
PANELS_DIR = OUT_ROOT / "panels"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PANELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    metrics_csv: Path
    images_dir: Path


METHODS = {
    "IDT": MethodSpec(
        "IDT",
        WORKSPACE / "SchrodingerBridge" / "docs" / "experiments" / "idt_eval_20260602" / "distinct5_512" / "idt_5x5" / "metrics.csv",
        WORKSPACE / "SchrodingerBridge" / "docs" / "experiments" / "idt_eval_20260602" / "distinct5_512" / "idt_5x5" / "images",
    ),
    "LBM-K": MethodSpec(
        "LBM-K",
        WORKSPACE / "SchrodingerBridge" / "exp" / "distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote" / "full_eval" / "epoch_0001" / "metrics.csv",
        WORKSPACE / "SchrodingerBridge" / "exp" / "distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote" / "full_eval" / "epoch_0001" / "images",
    ),
    "LBM-Knee": MethodSpec(
        "LBM-Knee",
        WORKSPACE / "SchrodingerBridge" / "aaai2027" / "local_eval" / "lbm_knee_e13_artfid" / "metrics.csv",
        WORKSPACE / "SchrodingerBridge" / "aaai2027" / "local_eval" / "lbm_knee_e13_artfid" / "images",
    ),
    "LBM-PS-v2": MethodSpec(
        "LBM-PS-v2",
        WORKSPACE / "SchrodingerBridge" / "aaai2027" / "local_eval" / "pattn_stokes002_e13" / "metrics.csv",
        WORKSPACE / "SchrodingerBridge" / "aaai2027" / "local_eval" / "pattn_stokes002_e13" / "images",
    ),
    "SaMST": MethodSpec(
        "SaMST",
        WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samst_distinct5_512_real_b2_e15_20260602" / "eval_epoch15" / "epoch_0015" / "metrics.csv",
        WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samst_distinct5_512_real_b2_e15_20260602" / "eval_epoch15" / "epoch_0015" / "images",
    ),
    "Seedream": MethodSpec(
        "Seedream",
        WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "seedream45_api" / "distinct5_512_seedream45_windhub_20260607_repaired750" / "metrics.csv",
        WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "seedream45_api" / "distinct5_512_seedream45_windhub_20260607_repaired750" / "images",
    ),
}

COMPARISONS = [
    ("LBM-Knee", "SaMST", "knee_vs_samst"),
    ("LBM-Knee", "Seedream", "knee_vs_seedream"),
    ("LBM-PS-v2", "SaMST", "psv2_vs_samst"),
    ("LBM-K", "IDT", "k_vs_idt"),
]

CELL = 176
PAD = 12
LABEL_H = 28


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
    ]
    for cand in candidates:
        p = Path(cand)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT = _font(18)
FONT_B = _font(20, bold=True)
FONT_S = _font(14)


def canon_src_name(src_style: str, src_image: str) -> str:
    prefix = f"{src_style}__"
    if src_image.startswith(prefix):
        return src_image[len(prefix) :]
    return src_image


def load_rows(spec: MethodSpec) -> dict[tuple[str, str, str], dict[str, str]]:
    with spec.metrics_csv.open("r", encoding="utf-8", newline="") as f:
        out = {}
        for row in csv.DictReader(f):
            key = (row["src_style"], row["tgt_style"], canon_src_name(row["src_style"], row["src_image"]))
            out[key] = row
        return out


def resolve_source(src_style: str, src_image: str) -> Path:
    direct = SOURCE_ROOT / src_style / src_image
    if direct.exists():
        return direct
    pref = SOURCE_ROOT / src_style / f"{src_style}__{src_image}"
    if pref.exists():
        return pref
    raise FileNotFoundError((src_style, src_image))


def resolve_target_ref(tgt_style: str) -> Path:
    items = sorted((SOURCE_ROOT / tgt_style).glob("*.jpg"))
    if not items:
        raise FileNotFoundError(tgt_style)
    return items[0]


def resolve_generated(spec: MethodSpec, row: dict[str, str]) -> Path:
    name = Path(str(row["gen_image"])).name
    p = spec.images_dir / name
    if p.exists():
        return p
    raw = spec.images_dir / str(row["gen_image"])
    if raw.exists():
        return raw
    raise FileNotFoundError(name)


def choose_cases(
    left_rows: dict[tuple[str, str, str], dict[str, str]],
    right_rows: dict[tuple[str, str, str], dict[str, str]],
    *,
    left_name: str,
    right_name: str,
    limit: int,
) -> list[tuple[float, tuple[str, str, str]]]:
    common = sorted(set(left_rows) & set(right_rows))
    scored: list[tuple[float, tuple[str, str, str]]] = []
    for key in common:
        src_style, tgt_style, _ = key
        if src_style == tgt_style:
            continue
        l = left_rows[key]
        r = right_rows[key]
        lc, ll = float(l["clip_style"]), float(l["content_lpips"])
        rc, rl = float(r["clip_style"]), float(r["content_lpips"])
        if left_name == "LBM-Knee" and right_name == "SaMST":
            score = (lc - rc) + 0.75 * (rl - ll)
        elif left_name == "LBM-Knee" and right_name == "Seedream":
            score = (lc - rc) + (rl - ll)
        elif left_name == "LBM-PS-v2" and right_name == "SaMST":
            score = (lc - rc) - 0.15 * max(ll - rl, 0.0)
        elif left_name == "LBM-K" and right_name == "IDT":
            score = (lc - rc) + 0.5 * (rl - ll)
        else:
            score = (lc - rc) + (rl - ll)
        scored.append((score, key))
    scored.sort(reverse=True)
    return scored[:limit]


def make_panel(
    *,
    source_path: Path,
    target_ref_path: Path,
    left_path: Path,
    right_path: Path,
    label_a: str,
    label_b: str,
    title: str,
    out_path: Path,
) -> None:
    width = 4 * CELL + 5 * PAD
    height = LABEL_H + CELL + 2 * PAD
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 4), title, fill=(20, 20, 20), font=FONT_B)

    cols = [
        ("Source", source_path),
        (label_a, left_path),
        (label_b, right_path),
        ("Target ref", target_ref_path),
    ]
    for idx, (label, path) in enumerate(cols):
        x0 = PAD + idx * (CELL + PAD)
        draw.text((x0 + CELL // 2, LABEL_H - 8), label, fill=(30, 30, 30), font=FONT_S, anchor="ma")
        with Image.open(path) as img:
            tile = img.convert("RGB").resize((CELL, CELL), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x0, LABEL_H))
        draw.rectangle([x0, LABEL_H, x0 + CELL, LABEL_H + CELL], outline=(180, 180, 180), width=1)
    canvas.save(out_path)


def main() -> None:
    rng = random.Random(20260608)
    rows_by_method = {name: load_rows(spec) for name, spec in METHODS.items()}
    manifest_rows = []
    case_id = 0
    for left_name, right_name, slug in COMPARISONS:
        chosen = choose_cases(rows_by_method[left_name], rows_by_method[right_name], left_name=left_name, right_name=right_name, limit=6)
        for _, key in chosen:
            src_style, tgt_style, src_image = key
            src_path = resolve_source(src_style, src_image)
            target_ref = resolve_target_ref(tgt_style)
            left_row = rows_by_method[left_name][key]
            right_row = rows_by_method[right_name][key]
            left_img = resolve_generated(METHODS[left_name], left_row)
            right_img = resolve_generated(METHODS[right_name], right_row)

            ab = [("A", left_name, left_img), ("B", right_name, right_img)]
            rng.shuffle(ab)
            panel_name = f"{case_id:03d}_{slug}_{src_style}_to_{tgt_style}.png"
            panel_path = PANELS_DIR / panel_name
            make_panel(
                source_path=src_path,
                target_ref_path=target_ref,
                left_path=ab[0][2],
                right_path=ab[1][2],
                label_a="Candidate A",
                label_b="Candidate B",
                title=f"{src_style} -> {tgt_style}",
                out_path=panel_path,
            )
            manifest_rows.append(
                {
                    "case_id": case_id,
                    "comparison": slug,
                    "src_style": src_style,
                    "tgt_style": tgt_style,
                    "src_image": src_image,
                    "panel_path": str(panel_path),
                    "candidate_a_method": ab[0][1],
                    "candidate_b_method": ab[1][1],
                    "source_path": str(src_path),
                    "target_ref_path": str(target_ref),
                }
            )
            case_id += 1

    manifest_csv = OUT_ROOT / "blind_pairwise_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_json = OUT_ROOT / "blind_pairwise_manifest.json"
    manifest_json.write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rubric_md = OUT_ROOT / "blind_pairwise_rubric.md"
    rubric_md.write_text(
        "\n".join(
            [
                "# Blind Pairwise Rubric",
                "",
                "For each panel, compare Candidate A and Candidate B under three blind questions:",
                "",
                "1. Which candidate better matches the target style?",
                "2. Which candidate better preserves the source content and structure?",
                "3. Which candidate has fewer artifacts / less muddy or grainy failure?",
                "",
                "Rate each question as one of:",
                "",
                "- A better",
                "- B better",
                "- Tie",
                "",
                "The evaluator should not see the method names during scoring.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(manifest_csv)
    print(manifest_json)
    print(rubric_md)


if __name__ == "__main__":
    main()
