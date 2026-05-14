"""S2WAT 750-image reproduction launcher.

This adapter uses the local S2WAT repository and already-trained per-style
checkpoints to generate the strict 5x5x30 protocol outputs. It writes images
with the same filenames as the SchrodingerBridge reference folder, records
inference timing, and can run the run_511 evaluation pack.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.utils import save_image


THIS_DIR = Path(__file__).resolve().parent
RUN511_ROOT = THIS_DIR.parent
RELATED_ROOT = RUN511_ROOT.parent
WORKSPACE_ROOT = RELATED_ROOT.parent
S2WAT_REPO = RELATED_ROOT / "repos" / "S2WAT-main"
STYLE_DATA = WORKSPACE_ROOT / "style_data"
OVERFIT50 = STYLE_DATA / "overfit50"
TRAIN_DATA = STYLE_DATA / "train"
DEFAULT_CHECKPOINT_ROOT = RELATED_ROOT / "baseline_pipeline" / "checkpoints" / "s2wat"
DEFAULT_REFERENCE_IMAGES = (
    WORKSPACE_ROOT
    / "SchrodingerBridge"
    / "exp"
    / "pareto_probe_4"
    / "S-add__K-3_C-2_W-10_Col-15"
    / "full_eval"
    / "epoch_0001"
    / "images"
)
STYLES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

PROFILES = {
    "4g": {"batch_size": 1, "train_images_per_style": 16, "epochs": 200, "img_size": 224, "train_size": 224},
    "7g": {"batch_size": 1, "train_images_per_style": 32, "epochs": 1000, "img_size": 256, "train_size": 224},
    "11g": {"batch_size": 2, "train_images_per_style": 64, "epochs": 2000, "img_size": 256, "train_size": 256},
}

EVAL_TASKS = {
    "base": ("eval_750.py", "eval_protocol750_sbmatch.json"),
    "guard": ("eval_guard_750.py", "eval_guard750.json"),
    "artifact": ("eval_artifact_pack_750.py", "eval_artifact_pack750.json"),
    "hf_kid": ("eval_hf_patch_kid_750.py", "eval_hf_patch_kid750.json"),
    "plain_kid": ("eval_plain_kid_750.py", "eval_plain_kid750.json"),
}


def _add_s2wat_to_path() -> None:
    sys.path.insert(0, str(S2WAT_REPO))


def reference_names(reference_images_dir: Path) -> list[str]:
    if reference_images_dir.is_dir():
        return sorted(p.name for p in reference_images_dir.iterdir() if p.is_file() and "_to_" in p.stem)
    names: list[str] = []
    for src_style in STYLES:
        for img in sorted((OVERFIT50 / src_style).glob("*.jpg"))[:30]:
            for target in STYLES:
                names.append(f"{src_style}_{img.stem}_to_{target}.jpg")
    return names


def parse_protocol_name(name: str) -> tuple[str, str, str]:
    stem = Path(name).stem
    if "_to_" not in stem:
        raise ValueError(f"Not a protocol output filename: {name}")
    prefix, target = stem.rsplit("_to_", 1)
    src_style, src_stem = prefix.split("_", 1)
    return src_style, src_stem, target


def style_reference_for(target_style: str) -> Path:
    candidates = sorted(p for p in (OVERFIT50 / target_style).iterdir() if p.suffix.lower() in IMG_EXTS)
    if not candidates:
        raise FileNotFoundError(f"No style reference images found for {target_style}: {OVERFIT50 / target_style}")
    return candidates[0]


def newest_checkpoint(style: str, checkpoint_root: Path) -> Path:
    style_dir = checkpoint_root / style
    candidates = sorted(
        style_dir.glob("checkpoint_*_epoch.pkl"),
        key=lambda p: int(p.stem.replace("checkpoint_", "").replace("_epoch", "")),
    )
    if not candidates:
        raise FileNotFoundError(f"No S2WAT checkpoint for {style}: {style_dir}")
    return candidates[-1]


def build_model(img_size: int, device: torch.device):
    _add_s2wat_to_path()
    from model.configuration import TransModule_Config
    from model.s2wat import S2WAT
    from net import Decoder_MVGG, TransModule
    from tools import Sample_Test_Net

    trans_cfg = TransModule_Config(
        nlayer=3,
        d_model=768,
        nhead=8,
        mlp_ratio=4,
        qkv_bias=False,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_first=True,
    )
    encoder = S2WAT(
        img_size=img_size,
        patch_size=2,
        in_chans=3,
        embed_dim=192,
        depths=[2, 2, 2],
        nhead=[3, 6, 12],
        strip_width=[2, 4, 7],
        drop_path_rate=0.0,
        patch_norm=True,
    )
    decoder = Decoder_MVGG(d_model=768, seq_input=True)
    trans = TransModule(trans_cfg)
    net = Sample_Test_Net(encoder, decoder, trans).to(device).eval()
    return net


def load_checkpoint(net, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    net.encoder.load_state_dict(checkpoint["encoder"])
    net.decoder.load_state_dict(checkpoint["decoder"])
    net.transModule.load_state_dict(checkpoint["transModule"])


def infer(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, Any]:
    _add_s2wat_to_path()
    from tools import content_style_transTo_pt

    if not S2WAT_REPO.is_dir():
        raise FileNotFoundError(f"S2WAT repo not found: {S2WAT_REPO}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    img_size = int(args.img_size or profile["img_size"])
    output_dir = args.run_root / "infer_750" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    names = reference_names(args.reference_images_dir)
    if args.limit_per_target > 0:
        limited: list[str] = []
        for target in STYLES:
            limited.extend([n for n in names if n.endswith(f"_to_{target}.jpg")][: args.limit_per_target])
        names = sorted(limited)

    start_all = time.perf_counter()
    per_target: list[dict[str, Any]] = []
    total_new = 0
    total_seen = 0

    for target in STYLES:
        target_names = [n for n in names if n.endswith(f"_to_{target}.jpg")]
        if not target_names:
            continue
        ckpt_path = newest_checkpoint(target, args.checkpoint_root)
        style_path = style_reference_for(target)
        net = build_model(img_size, device)
        load_checkpoint(net, ckpt_path, device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        generated = 0
        skipped = 0
        with torch.no_grad():
            for out_name in target_names:
                out_path = output_dir / out_name
                total_seen += 1
                if out_path.exists() and not args.force:
                    skipped += 1
                    continue
                src_style, src_stem, _ = parse_protocol_name(out_name)
                content_path = OVERFIT50 / src_style / f"{src_stem}.jpg"
                if not content_path.exists():
                    raise FileNotFoundError(f"Missing content image for {out_name}: {content_path}")
                i_c, i_s = content_style_transTo_pt(str(content_path), str(style_path))
                output = net(i_c.to(device), i_s.to(device), arbitrary_input=True)
                save_image(output.detach().cpu(), str(out_path))
                generated += 1
                total_new += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        per_target.append(
            {
                "target": target,
                "checkpoint": str(ckpt_path),
                "style_reference": str(style_path),
                "images_requested": len(target_names),
                "generated": generated,
                "skipped_existing": skipped,
                "elapsed_sec": round(elapsed, 3),
                "sec_per_requested": round(elapsed / max(1, len(target_names)), 6),
            }
        )
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - start_all
    image_count = len(list(output_dir.glob("*.jpg")))
    status = "ok" if (args.limit_per_target > 0 or image_count >= 750) else "partial"
    return {
        "stage": "infer",
        "status": status,
        "device": str(device),
        "img_size": img_size,
        "images_dir": str(output_dir),
        "images": image_count,
        "requested": total_seen,
        "generated_new": total_new,
        "elapsed_sec": round(total_elapsed, 3),
        "sec_per_requested": round(total_elapsed / max(1, total_seen), 6),
        "per_target": per_target,
    }


def train(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, Any]:
    vgg_path = S2WAT_REPO / "pre_trained_models" / "vgg_normalised.pth"
    if not vgg_path.exists():
        raise FileNotFoundError(f"Missing VGG weights: {vgg_path}")

    styles = args.styles or STYLES
    epochs = int(args.epochs or profile["epochs"])
    batch_size = int(args.batch_size or profile["batch_size"])
    img_size = int(args.img_size or profile["img_size"])
    train_size = int(args.train_size if args.train_size >= 0 else profile["train_size"])
    ckpt_root = args.run_root / "checkpoints" / "s2wat"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    rows = []
    for style in styles:
        content_dir = TRAIN_DATA / "photo"
        style_dir = TRAIN_DATA / style
        save_dir = ckpt_root / style
        save_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(S2WAT_REPO / "train.py"),
            "--content_dir",
            str(content_dir),
            "--style_dir",
            str(style_dir),
            "--vgg_dir",
            str(vgg_path),
            "--epoch",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--img_size",
            str(img_size),
            "--train_size",
            str(train_size),
            "--precision",
            args.precision,
            "--grad_checkpoint",
            "--checkpoint_save_interval",
            str(max(1, epochs)),
            "--loss_count_interval",
            str(max(1, min(50, epochs))),
            "--checkpoint_save_path",
            str(save_dir),
        ]
        t0 = time.perf_counter()
        print(f"[S2WAT TRAIN] style={style} epochs={epochs} batch={batch_size}", flush=True)
        proc = subprocess.run(cmd, cwd=str(S2WAT_REPO))
        rows.append({"style": style, "returncode": proc.returncode, "elapsed_sec": round(time.perf_counter() - t0, 3)})
        if proc.returncode != 0:
            return {"stage": "train", "status": "failed", "per_style": rows, "elapsed_sec": round(time.perf_counter() - start, 3)}

    return {
        "stage": "train",
        "status": "ok",
        "checkpoint_root": str(ckpt_root),
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "train_size": train_size,
        "elapsed_sec": round(time.perf_counter() - start, 3),
        "per_style": rows,
    }


def run_eval_tasks(args: argparse.Namespace) -> dict[str, Any]:
    images_dir = args.run_root / "infer_750" / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    rows = []
    start = time.perf_counter()
    for task in args.eval_tasks:
        script, out_name = EVAL_TASKS[task]
        output = images_dir.parent / out_name
        cmd = [
            sys.executable,
            str(RUN511_ROOT / "eval" / script),
            "--images_dir",
            str(images_dir),
            "--output",
            str(output),
        ]
        if task in {"base", "guard", "artifact"}:
            cmd.extend(["--max_ref_cache", str(args.max_ref_cache)])
        t0 = time.perf_counter()
        print(f"[S2WAT EVAL] {task} -> {output}", flush=True)
        proc = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT))
        rows.append(
            {
                "task": task,
                "returncode": proc.returncode,
                "output": str(output),
                "elapsed_sec": round(time.perf_counter() - t0, 3),
                "status": "ok" if proc.returncode == 0 and output.exists() else "failed",
            }
        )
        if proc.returncode != 0:
            break
    return {"stage": "eval", "status": "ok" if all(r["status"] == "ok" for r in rows) else "failed", "elapsed_sec": round(time.perf_counter() - start, 3), "tasks": rows}


def check(args: argparse.Namespace) -> dict[str, Any]:
    refs = reference_names(args.reference_images_dir)
    ckpts = {style: str(newest_checkpoint(style, args.checkpoint_root)) for style in STYLES}
    return {
        "stage": "check",
        "status": "ok",
        "repo": str(S2WAT_REPO),
        "checkpoint_root": str(args.checkpoint_root),
        "reference_images": len(refs),
        "checkpoints": ckpts,
    }


def write_summary(run_root: Path, rows: list[dict[str, Any]]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "summary.json"
    existing: list[dict[str, Any]] = []
    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            existing = list(loaded.get("runs", []))
        except Exception:
            existing = []
    by_stage = {str(row.get("stage", f"row_{idx}")): row for idx, row in enumerate(existing)}
    for row in rows:
        by_stage[str(row.get("stage", f"row_{len(by_stage)}"))] = row
    merged = list(by_stage.values())
    summary_path.write_text(json.dumps({"runs": merged}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    flat_rows = []
    for row in merged:
        flat_rows.append({k: v for k, v in row.items() if not isinstance(v, (list, dict))})
    if flat_rows:
        keys = sorted({k for row in flat_rows for k in row})
        with (run_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat_rows)


def parse_styles(value: str) -> list[str] | None:
    if not value:
        return None
    styles = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in styles if s not in STYLES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown styles: {unknown}; valid={STYLES}")
    return styles


def main() -> int:
    parser = argparse.ArgumentParser(description="Run S2WAT strict protocol-750 inference and evaluation.")
    parser.add_argument("--mode", choices=["check", "train", "infer", "eval", "all", "smoke"], default="all")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="7g")
    parser.add_argument("--run_root", type=Path, default=RUN511_ROOT / "outputs" / "s2wat_750_strict")
    parser.add_argument("--checkpoint_root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--reference_images_dir", type=Path, default=DEFAULT_REFERENCE_IMAGES)
    parser.add_argument("--styles", type=parse_styles, default=None, help="Comma-separated subset for training.")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--precision", choices=["fp32", "amp", "bf16"], default="bf16")
    parser.add_argument("--limit_per_target", type=int, default=0, help="0 means full 150 per target / 750 total.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing output images.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--eval_tasks", nargs="*", choices=sorted(EVAL_TASKS), default=["base", "guard"])
    parser.add_argument("--max_ref_cache", type=int, default=256)
    args = parser.parse_args()

    args.run_root = args.run_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.reference_images_dir = args.reference_images_dir.resolve()

    if args.mode == "smoke":
        args.mode = "all"
        args.run_root = (RUN511_ROOT / "outputs" / "s2wat_smoke").resolve()
        args.limit_per_target = 1
        args.eval_tasks = ["base"]

    rows: list[dict[str, Any]] = []
    try:
        if args.mode == "check":
            rows.append(check(args))
        elif args.mode == "train":
            rows.append(train(args, PROFILES[args.profile]))
        elif args.mode == "infer":
            rows.append(infer(args, PROFILES[args.profile]))
        elif args.mode == "eval":
            rows.append(run_eval_tasks(args))
        elif args.mode == "all":
            rows.append(check(args))
            rows.append(infer(args, PROFILES[args.profile]))
            if rows[-1]["status"] in {"ok", "partial"}:
                rows.append(run_eval_tasks(args))
        else:
            raise AssertionError(args.mode)
    except Exception as exc:
        rows.append({"stage": args.mode, "status": "failed", "error": repr(exc)})
        write_summary(args.run_root, rows)
        print(f"[S2WAT ERROR] {exc}", flush=True)
        return 1

    write_summary(args.run_root, rows)
    print(f"[S2WAT SUMMARY] {args.run_root / 'summary.json'}", flush=True)
    return 0 if rows and rows[-1].get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
