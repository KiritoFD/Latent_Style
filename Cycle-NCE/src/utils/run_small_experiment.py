from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s.strip())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "x"


def _method_signature(cfg: Dict, cfg_path: Path) -> str:
    model = cfg.get("model", {})
    loss = cfg.get("loss", {})
    parts = [
        _slug(cfg_path.stem),
        f"bd{int(model.get('base_dim', 0) or 0)}",
        f"dsp{1 if bool(model.get('use_decoder_spatial_inject', False)) else 0}",
        f"hp{str(model.get('style_delta_lowfreq_gain', 'na')).replace('.', 'p')}",
        f"whf{str(loss.get('w_featmatch_hf', 0)).replace('.', 'p')}",
        f"wprob{str(loss.get('w_prob', 0)).replace('.', 'p')}",
        f"wproto{str(loss.get('w_proto', 0)).replace('.', 'p')}",
        f"wcyc{str(loss.get('w_cycle', 0)).replace('.', 'p')}",
    ]
    raw = "-".join(parts)
    return raw[:120]


def _build_collage(
    metrics_csv: Path,
    eval_dir: Path,
    test_dir: Path,
    style_names: List[str],
    out_path: Path,
    per_pair: int = 3,
    image_size: int = 256,
) -> int:
    if not metrics_csv.exists():
        return 0

    valid_pairs = {(s, t) for s in style_names for t in style_names}
    rows: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for pair in valid_pairs:
        rows[pair] = []

    with open(metrics_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            src = str(r.get("src_style", "")).strip()
            tgt = str(r.get("tgt_style", "")).strip()
            if (src, tgt) not in valid_pairs:
                continue
            if len(rows[(src, tgt)]) >= per_pair:
                continue
            src_name = str(r.get("src_image", "")).strip()
            gen_name = str(r.get("gen_image", "")).strip()
            src_path = test_dir / src / src_name
            gen_path = eval_dir / gen_name
            if not src_path.exists() or not gen_path.exists():
                continue
            rows[(src, tgt)].append(
                {
                    "src": str(src_path),
                    "gen": str(gen_path),
                    "caption": f"{src}->{tgt}",
                }
            )

    flat: List[Dict[str, str]] = []
    for pair in sorted(rows.keys()):
        flat.extend(rows[pair])
    if not flat:
        return 0

    cell_w = image_size * 2 + 24
    cell_h = image_size + 36
    canvas = Image.new("RGB", (cell_w, cell_h * len(flat)), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    for i, item in enumerate(flat):
        y0 = i * cell_h
        draw.text((8, y0 + 6), item["caption"], fill=(230, 230, 230))
        src = Image.open(item["src"]).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        gen = Image.open(item["gen"]).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        canvas.paste(src, (8, y0 + 28))
        canvas.paste(gen, (12 + image_size, y0 + 28))
        draw.text((8, y0 + 28 + image_size + 4), "src", fill=(170, 170, 170))
        draw.text((12 + image_size, y0 + 28 + image_size + 4), "gen", fill=(170, 170, 170))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    return len(flat)


def _append_experiment_doc(
    doc_path: Path,
    name: str,
    config_path: Path,
    summary_path: Path,
    collage_path: Path,
) -> None:
    summary = _load_json(summary_path)
    transfer = summary.get("analysis", {}).get("style_transfer_ability", {})
    p2a = summary.get("analysis", {}).get("photo_to_art_performance", {})
    cond = summary.get("analysis", {}).get("conditional_sensitivity", {})
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"## {name} ({timestamp})",
        f"- config: `{config_path}`",
        f"- summary: `{summary_path}`",
        f"- collage: `{collage_path}`",
        f"- transfer clip_style: `{transfer.get('clip_style')}`",
        f"- transfer content_lpips: `{transfer.get('content_lpips')}`",
        f"- transfer classifier_acc: `{transfer.get('classifier_acc')}`",
        f"- photo_to_art clip_style: `{p2a.get('clip_style')}`",
        f"- photo_to_art classifier_acc: `{p2a.get('classifier_acc')}`",
        f"- cond pair_count: `{cond.get('pair_count')}`",
        f"- cond delta_abs: `{cond.get('delta_abs')}`",
        f"- cond delta_high_ratio: `{cond.get('delta_high_ratio')}`",
        "",
    ]
    with open(doc_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short experiment with eval collage and log.")
    parser.add_argument("--config", type=str, default="../config.json")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max_src_samples", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--per_pair", type=int, default=3)
    args = parser.parse_args()

    src_root = Path(__file__).resolve().parents[1]
    repo_root = src_root.parent
    base_cfg_path = (src_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = _load_json(base_cfg_path)
    base_cfg_root = base_cfg_path.parent

    ts = time.strftime("%Y%m%d_%H%M%S")
    method_sig = _method_signature(cfg, base_cfg_path)
    save_dir = f"../small-exp-{_slug(args.name)}-{method_sig}-{ts}"
    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"]["save_dir"] = save_dir

    cfg.setdefault("training", {})
    t = cfg["training"]
    t["resume_checkpoint"] = ""
    t["num_epochs"] = int(args.epochs)
    t["save_interval"] = max(1, int(args.epochs))
    t["full_eval_interval"] = max(1, int(args.epochs))
    t["full_eval_on_last_epoch"] = True
    t["full_eval_max_src_samples"] = int(args.max_src_samples)
    t["full_eval_save_images"] = True
    t["batch_size"] = int(args.batch_size)
    t["use_compile"] = False
    t["auto_preload_latents_to_gpu"] = True
    t["auto_preload_gpu_budget_mb"] = 2048
    t["run_lock_path"] = f"../run_{_slug(args.name)}.lock"

    cfg.setdefault("loss", {})
    cls_ckpt_raw = str(cfg["loss"].get("style_classifier_ckpt", "")).strip()
    if cls_ckpt_raw:
        cls_path = Path(cls_ckpt_raw)
        if not cls_path.is_absolute():
            cls_path = (base_cfg_root / cls_path).resolve()
        cfg["loss"]["style_classifier_ckpt"] = str(cls_path)

    cfg.setdefault("data", {})
    data_root_raw = str(cfg["data"].get("data_root", "")).strip()
    if data_root_raw:
        data_root = Path(data_root_raw)
        if not data_root.is_absolute():
            data_root = (base_cfg_root / data_root).resolve()
        cfg["data"]["data_root"] = str(data_root)

    test_dir_raw = str(cfg.get("training", {}).get("test_image_dir", "")).strip()
    if test_dir_raw:
        test_dir = Path(test_dir_raw)
        if not test_dir.is_absolute():
            test_dir = (base_cfg_root / test_dir).resolve()
        cfg["training"]["test_image_dir"] = str(test_dir)

    exp_cfg_dir = src_root / "experiments"
    exp_cfg_path = exp_cfg_dir / f"{args.name}.json"
    _save_json(exp_cfg_path, cfg)

    cmd = [sys.executable, "run.py", "--config", str(exp_cfg_path)]
    subprocess.run(cmd, cwd=str(src_root), check=True)

    ckpt_dir = (src_root / save_dir).resolve()
    eval_dir = ckpt_dir / "full_eval" / f"epoch_{int(args.epochs):04d}"
    summary_path = eval_dir / "summary.json"
    metrics_csv = eval_dir / "metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    style_names = list(cfg.get("data", {}).get("style_subdirs", []))
    test_dir_raw = str(cfg.get("training", {}).get("test_image_dir", ""))
    test_dir = Path(test_dir_raw).resolve()
    collage_path = eval_dir / "collage.jpg"
    used = _build_collage(
        metrics_csv=metrics_csv,
        eval_dir=eval_dir,
        test_dir=test_dir,
        style_names=style_names,
        out_path=collage_path,
        per_pair=max(1, int(args.per_pair)),
    )
    print(f"collage_rows={used} -> {collage_path}")

    doc_path = repo_root / "EXPERIMENTS.md"
    _append_experiment_doc(
        doc_path=doc_path,
        name=args.name,
        config_path=exp_cfg_path,
        summary_path=summary_path,
        collage_path=collage_path,
    )
    print(f"logged -> {doc_path}")


if __name__ == "__main__":
    main()
