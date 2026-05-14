"""High-frequency patch KID diagnostics for protocol-750 outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
OVERFIT50 = WORKSPACE_ROOT / "style_data" / "overfit50"
STYLES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_name(path: Path) -> tuple[str, str, str] | None:
    if "_to_" not in path.stem:
        return None
    prefix, target = path.stem.rsplit("_to_", 1)
    if "_" not in prefix:
        return None
    src_style, stem = prefix.split("_", 1)
    return src_style, stem, target


def load_rgb(path: Path, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img).astype("float32") / 255.0


def highpass_rgb(rgb: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    hp = rgb - blur
    hp = hp - hp.min()
    hp = hp / max(float(hp.max()), 1e-8)
    return hp


def extract_patches(rgb: np.ndarray, patch: int = 32, stride: int = 32, max_patches: int = 32) -> list[np.ndarray]:
    hp = highpass_rgb(rgb)
    patches = []
    for y in range(0, hp.shape[0] - patch + 1, stride):
        for x in range(0, hp.shape[1] - patch + 1, stride):
            patches.append(hp[y:y + patch, x:x + patch, :])
    if len(patches) > max_patches:
        idx = np.linspace(0, len(patches) - 1, max_patches, dtype=int)
        patches = [patches[i] for i in idx]
    return patches


class VGGPatchFeat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights).features[:16]
        self.features = vgg.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.tf = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
            ]
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        feat = self.features(batch)
        feat = feat.mean(dim=(2, 3))
        return feat

    def encode_patches(self, patches: list[np.ndarray], device: torch.device) -> np.ndarray:
        if not patches:
            return np.zeros((0, 256), dtype=np.float32)
        tensors = [self.tf(Image.fromarray(np.clip(p * 255.0, 0, 255).astype("uint8"))) for p in patches]
        feats = []
        with torch.no_grad():
            for start in range(0, len(tensors), 64):
                batch = torch.stack(tensors[start:start + 64]).to(device)
                feats.append(self.forward(batch).cpu().numpy())
        return np.concatenate(feats, axis=0)


def polynomial_mmd_2(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float | None = None, coef0: float = 1.0) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    k_xx = (gamma * (x @ x.T) + coef0) ** degree
    k_yy = (gamma * (y @ y.T) + coef0) ** degree
    k_xy = (gamma * (x @ y.T) + coef0) ** degree
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    m = x.shape[0]
    n = y.shape[0]
    term_xx = k_xx.sum() / (m * (m - 1))
    term_yy = k_yy.sum() / (n * (n - 1))
    term_xy = 2.0 * k_xy.mean()
    return float(term_xx + term_yy - term_xy)


def collect_target_features(
    paths: list[Path],
    encoder: VGGPatchFeat,
    device: torch.device,
    max_images: int = 64,
) -> np.ndarray:
    feats = []
    for path in paths[:max_images]:
        rgb = load_rgb(path)
        patches = extract_patches(rgb)
        patch_feats = encoder.encode_patches(patches, device)
        if len(patch_feats):
            feats.append(patch_feats)
    if not feats:
        return np.zeros((0, 256), dtype=np.float32)
    return np.concatenate(feats, axis=0)


def eval_hf_patch_kid(images_dir: Path) -> dict[str, object]:
    encoder = VGGPatchFeat().to(DEVICE)
    device = torch.device(DEVICE)
    by_target: dict[str, list[Path]] = {target: [] for target in STYLES}
    for gen_path in sorted(images_dir.glob("*.jpg")):
        parsed = parse_name(gen_path)
        if parsed is None:
            continue
        target = parsed[2]
        if target in by_target:
            by_target[target].append(gen_path)

    rows = []
    for target in STYLES:
        gen_paths = by_target[target]
        style_paths = sorted((OVERFIT50 / target).glob("*.jpg"))
        gen_feats = collect_target_features(gen_paths, encoder, device, max_images=150)
        style_feats = collect_target_features(style_paths, encoder, device, max_images=50)
        kid = polynomial_mmd_2(gen_feats, style_feats)
        rows.append({"target": target, "images": len(gen_paths), "hf_patch_kid": round(kid, 6)})

    overall = {
        "target": "ALL",
        "images": sum(int(r["images"]) for r in rows),
        "hf_patch_kid": round(float(np.mean([float(r["hf_patch_kid"]) for r in rows])), 6),
    }
    return {"results": rows + [overall]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = eval_hf_patch_kid(args.images_dir.resolve())
    output = args.output or args.images_dir.parent / "eval_hf_patch_kid750.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "images", "hf_patch_kid"])
        writer.writeheader()
        writer.writerows(result["results"])

    print(output)
    print(csv_path)
    print(json.dumps(result["results"][-1], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
