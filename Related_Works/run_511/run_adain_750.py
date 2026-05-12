"""Self-contained AdaIN train + 750-image inference launcher.

AdaIN (Huang & Belongie 2017) is a feedforward style transfer method.
Training trains only the decoder; the encoder (VGG) is frozen.

This script lives in run_511 and does not import or reference Related_Works.
It uses:
  - run_511/repos/adain for model code
  - style_data/ for local training and evaluation content/style images
  - the Ours 750-image folder only as an optional filename manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent
REPO_DIR = THIS_DIR / "repos" / "adain"
STYLE_DATA = WORKSPACE_ROOT / "style_data"
TRAIN_DATA = STYLE_DATA / "train"
OVERFIT50 = STYLE_DATA / "overfit50"
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
    "4g": {"batch_size": 4, "train_images_per_style": 16, "max_iter": 16000},
    "7g": {"batch_size": 8, "train_images_per_style": 32, "max_iter": 32000},
    "11g": {"batch_size": 16, "train_images_per_style": 64, "max_iter": 160000},
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StyleTransferDataset(Dataset):
    """Pairs every content image with every style image."""

    def __init__(self, content_dir: Path, style_dir: Path, transform, max_per_style: int = 0):
        self.transform = transform
        self.content_paths = self._collect(content_dir, max_per_style)
        self.style_paths = self._collect(style_dir, max_per_style)
        self.pairs = [(c, s) for c in self.content_paths for s in self.style_paths]

    @staticmethod
    def _collect(root: Path, limit: int) -> list[Path]:
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
        return paths[:limit] if limit > 0 else paths

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        c_path, s_path = self.pairs[idx]
        content = self.transform(Image.open(c_path).convert("RGB"))
        style = self.transform(Image.open(s_path).convert("RGB"))
        return content, style


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_vgg_weights() -> Path:
    """Find vgg_normalised.pth from any baseline repo."""
    candidates = [
        THIS_DIR / "repos" / "AesFA" / "vgg_normalised.pth",
        THIS_DIR / "repos" / "StyTR-2" / "experiments" / "vgg_normalised.pth",
        THIS_DIR / "repos" / "cast" / "models" / "vgg_normalised.pth",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find vgg_normalised.pth in any known location")


def copy_images(src: Path, dst: Path, limit: int | None = None, prefix: str | None = None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(src.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        name = f"{prefix}_{img.name}" if prefix else img.name
        shutil.copy2(img, dst / name)
        count += 1
        if limit is not None and count >= limit:
            break
    return count


def prepare_train_data(work_dir: Path, images_per_style: int) -> tuple[Path, Path]:
    content_dir = work_dir / "train_content"
    style_dir = work_dir / "train_style"
    if content_dir.exists():
        shutil.rmtree(content_dir)
    if style_dir.exists():
        shutil.rmtree(style_dir)
    copy_images(TRAIN_DATA / "photo", content_dir, images_per_style)
    for style in STYLES:
        src = TRAIN_DATA / style
        if src.is_dir():
            copy_images(src, style_dir / style, images_per_style)
    return content_dir, style_dir


def reference_names(reference_images_dir: Path) -> list[str]:
    if not reference_images_dir.is_dir():
        names = []
        for src_style in STYLES:
            src_dir = OVERFIT50 / src_style
            for img in sorted(src_dir.glob("*.jpg"))[:30]:
                for target in STYLES:
                    names.append(f"{src_style}_{img.stem}_to_{target}.jpg")
        return names
    return sorted(p.name for p in reference_images_dir.iterdir() if p.is_file() and "_to_" in p.stem)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    sys.path.insert(0, str(REPO_DIR))
    from adain_net import AdaINNet

    work_dir = args.run_root / "work" / "adain"
    save_dir = args.run_root / "checkpoints" / "adain"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.run_root / "logs" / "adain_train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    images_per_style = int(args.train_images_per_style or profile["train_images_per_style"])
    max_iter = int(args.max_iter or profile["max_iter"])
    batch_size = int(args.batch_size or profile["batch_size"])

    content_dir, style_dir = prepare_train_data(work_dir, images_per_style)

    # Use standard VGG-19 ImageNet pretrained (not vgg_normalised.pth which
    # produces tiny features at relu4_1 and causes decoder collapse).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    dataset = StyleTransferDataset(content_dir, style_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = AdaINNet(vgg_weights=None).to(device)
    model.train()
    # Only train decoder
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)

    start = time.time()
    iteration = 0
    log_interval = max(100, max_iter // 100)  # log ~100 times during training
    msg0 = f"AdaIN training: max_iter={max_iter}, batch_size={batch_size}, device={device}, vgg=IMAGENET1K_V1"
    print(msg0, flush=True)
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(msg0 + "\n")
        logf.flush()

        while iteration < max_iter:
            for content, style in loader:
                if iteration >= max_iter:
                    break
                content = content.to(device)
                style = style.to(device)

                output = model(content, style)
                out_feats = model.encode(output)
                c_feats = model.encode(content)
                s_feats = model.encode(style)

                loss_c = model.calc_content_loss(out_feats, c_feats)
                loss_s = model.calc_style_loss(out_feats, s_feats)
                loss = loss_c + 10.0 * loss_s

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iteration += 1

                if iteration % log_interval == 0 or iteration == 1 or iteration == max_iter:
                    elapsed = time.time() - start
                    it_per_sec = iteration / elapsed if elapsed > 0 else 0
                    eta = (max_iter - iteration) / it_per_sec if it_per_sec > 0 else 0
                    msg = (f"iter {iteration}/{max_iter}  "
                           f"loss_c={loss_c.item():.6f}  loss_s={loss_s.item():.4f}  total={loss.item():.4f}  "
                           f"lr={optimizer.param_groups[0]['lr']:.6f}  "
                           f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  {it_per_sec:.1f} it/s")
                    print(msg, flush=True)
                    logf.write(msg + "\n")
                    logf.flush()

        # Save decoder weights
        ckpt_path = save_dir / "decoder.pth"
        torch.save(model.decoder.state_dict(), ckpt_path)
        logf.write(f"Saved checkpoint: {ckpt_path}\n")

    return {
        "stage": "train",
        "status": "ok",
        "returncode": 0,
        "elapsed_sec": round(time.time() - start, 3),
        "checkpoint_dir": str(save_dir),
        "log_path": str(log_path),
        "max_iter": max_iter,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Infer
# ---------------------------------------------------------------------------

def infer(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    sys.path.insert(0, str(REPO_DIR))
    from adain_net import AdaINNet

    max_iter = int(args.max_iter or profile["max_iter"])
    ckpt_path = args.run_root / "checkpoints" / "adain" / "decoder.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing AdaIN decoder checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AdaINNet(vgg_weights=None).to(device)
    model.decoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    output_dir = args.run_root / "infer_750" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = reference_names(args.reference_images_dir)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    ])

    rows = []
    start_all = time.time()
    total = 0

    with torch.no_grad():
        for target in STYLES:
            target_ref = [n for n in reference if n.endswith(f"_to_{target}.jpg")]
            if args.limit_per_target > 0:
                target_ref = target_ref[:args.limit_per_target]

            style_dir = OVERFIT50 / target
            first_style = next(iter(sorted(style_dir.glob("*.jpg"))), None)
            if first_style is None:
                rows.append({"target": target, "returncode": 1, "renamed": 0, "error": "no style image"})
                continue
            style_img = transform(Image.open(first_style).convert("RGB")).unsqueeze(0).to(device)

            start = time.time()
            renamed = 0
            for out_name in target_ref:
                prefix = out_name[: -len(f"_to_{target}.jpg")]
                src_style, stem = prefix.split("_", 1)
                src = OVERFIT50 / src_style / f"{stem}.jpg"
                if not src.exists():
                    continue
                content_img = transform(Image.open(src).convert("RGB")).unsqueeze(0).to(device)
                output = model(content_img, style_img)
                output = inv_transform(output.squeeze(0).clamp(0, 1))
                out_pil = transforms.ToPILImage()(output.cpu())
                out_pil.save(output_dir / out_name)
                renamed += 1

            total += renamed
            rows.append({
                "target": target,
                "returncode": 0,
                "renamed": renamed,
                "elapsed_sec": round(time.time() - start, 3),
            })

    status = "ok"
    if args.limit_per_target == 0 and total != 750:
        status = "partial" if total > 0 else "failed"
    return {
        "stage": "infer",
        "status": status,
        "elapsed_sec": round(time.time() - start_all, 3),
        "images": total,
        "images_dir": str(output_dir),
        "per_target": rows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_summary(run_root: Path, rows: list[dict[str, object]]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "summary.json").write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (run_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        keys = sorted({k for row in rows for k in row.keys() if k != "per_target"})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "all", "smoke"], default="all")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="7g")
    parser.add_argument("--run_root", type=Path, default=THIS_DIR / "outputs" / "adain_750")
    parser.add_argument("--reference_images_dir", type=Path, default=DEFAULT_REFERENCE_IMAGES)
    parser.add_argument("--max_iter", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--train_images_per_style", type=int, default=0)
    parser.add_argument("--limit_per_target", type=int, default=0, help="0 means full 150 per target / 750 total.")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.reference_images_dir = args.reference_images_dir.resolve()
    profile = PROFILES[args.profile]
    if args.mode == "smoke":
        args.max_iter = 1
        args.batch_size = 2
        args.train_images_per_style = 2
        args.limit_per_target = 1
        args.mode = "all"

    rows: list[dict[str, object]] = []
    if args.mode in {"train", "all"}:
        rows.append(train(args, profile))
        write_summary(args.run_root, rows)
        if rows[-1]["status"] != "ok":
            return 1
    if args.mode in {"infer", "all"}:
        rows.append(infer(args, profile))
        write_summary(args.run_root, rows)
        if rows[-1]["status"] != "ok":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
