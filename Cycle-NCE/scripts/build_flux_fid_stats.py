from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class InceptionFeatRunner:
    def __init__(self, device: str = "cuda", batch_size: int = 32):
        self.device = device
        self.batch_size = max(1, int(batch_size))
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval().to(self.device)
        self.model = model
        self.tf = T.Compose(
            [
                T.Resize((299, 299)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def extract(self, paths: list[Path], max_images: int = 5000) -> np.ndarray:
        sel = paths[: max(1, int(max_images))]
        if not sel:
            return np.zeros((0, 2048), dtype=np.float64)
        feats = []
        with torch.no_grad():
            for i in range(0, len(sel), self.batch_size):
                batch_paths = sel[i : i + self.batch_size]
                imgs = []
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        imgs.append(self.tf(img))
                    except Exception:
                        continue
                if not imgs:
                    continue
                x = torch.stack(imgs, dim=0).to(self.device)
                y = self.model(x)
                feats.append(y.detach().cpu().numpy().astype(np.float64))
        if not feats:
            return np.zeros((0, 2048), dtype=np.float64)
        return np.concatenate(feats, axis=0)


def resolve_path(raw: str, bases: list[Path]) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    for b in bases:
        cand = (b / p).resolve()
        if cand.exists():
            return cand
    return (bases[0] / p).resolve()


def list_images(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    out = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    out.sort(key=lambda x: x.name)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-style FLUX proxy FID stats (*.npz with mu/sigma).")
    parser.add_argument("--config", type=str, default="Cycle-NCE/src/config.json")
    parser.add_argument("--flux_root", type=str, required=True, help="Root dir of FLUX images grouped by style subdir.")
    parser.add_argument("--out_dir", type=str, default="Cycle-NCE/artifacts/fid_proxy_stats")
    parser.add_argument("--styles", type=str, default="", help="Comma-separated styles override; default from config.data.style_subdirs.")
    parser.add_argument("--exclude_styles", type=str, default="photo", help="Comma-separated styles to skip.")
    parser.add_argument("--max_images", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    here = Path(__file__).resolve()
    bases = [Path.cwd(), here.parent, here.parents[1], here.parents[2]]
    cfg_path = resolve_path(args.config, bases)
    flux_root = resolve_path(args.flux_root, bases)
    out_dir = resolve_path(args.out_dir, bases)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    default_styles = list(cfg.get("data", {}).get("style_subdirs", []))
    test_image_dir = str(cfg.get("training", {}).get("test_image_dir", ""))

    if args.styles.strip():
        styles = [x.strip() for x in args.styles.split(",") if x.strip()]
    else:
        styles = default_styles
    exclude = {x.strip().lower() for x in args.exclude_styles.split(",") if x.strip()}
    styles = [s for s in styles if s.lower() not in exclude]

    print(f"[CONFIG] {cfg_path}")
    print(f"[CONFIG] test_image_dir={test_image_dir}")
    print(f"[FLUX_ROOT] {flux_root}")
    print(f"[OUT_DIR] {out_dir}")
    print(f"[STYLES] {styles}")

    runner = InceptionFeatRunner(device=args.device, batch_size=args.batch_size)
    for style in styles:
        style_dir = flux_root / style
        img_paths = list_images(style_dir)
        if not img_paths:
            print(f"[SKIP] no images for style={style} at {style_dir}")
            continue
        feats = runner.extract(img_paths, max_images=args.max_images)
        if feats.shape[0] < 2:
            print(f"[SKIP] insufficient images for style={style}, n={feats.shape[0]}")
            continue
        mu = feats.mean(axis=0)
        sigma = np.cov(feats, rowvar=False)
        out_path = out_dir / f"{style}_stats.npz"
        np.savez_compressed(out_path, mu=mu, sigma=sigma, n=int(feats.shape[0]), style=style, src_dir=str(style_dir))
        print(f"[OK] style={style} n={feats.shape[0]} -> {out_path}")


if __name__ == "__main__":
    main()
