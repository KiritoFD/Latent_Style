"""Plain image-level KID for protocol-750 outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


class InceptionFeat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        net = models.inception_v3(weights=weights, transform_input=False)
        net.fc = nn.Identity()
        net.eval()
        self.net = net
        for p in self.parameters():
            p.requires_grad_(False)
        self.tf = T.Compose(
            [
                T.Resize((299, 299)),
                T.ToTensor(),
                T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
            ]
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.net(batch)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def encode_paths(self, paths: list[Path], device: torch.device, batch_size: int = 32) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start:start + batch_size]
                batch = torch.stack([self.tf(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
                feats.append(self.forward(batch).cpu().numpy())
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 2048), dtype=np.float32)


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


def eval_plain_kid(images_dir: Path) -> dict[str, object]:
    encoder = InceptionFeat().to(DEVICE)
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
        gen_feats = encoder.encode_paths(gen_paths, device)
        style_feats = encoder.encode_paths(style_paths, device)
        kid = polynomial_mmd_2(gen_feats, style_feats)
        rows.append({"target": target, "images": len(gen_paths), "kid": round(kid, 6)})

    overall = {
        "target": "ALL",
        "images": sum(int(r["images"]) for r in rows),
        "kid": round(float(np.mean([float(r["kid"]) for r in rows])), 6),
    }
    return {"results": rows + [overall]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = eval_plain_kid(args.images_dir.resolve())
    output = args.output or args.images_dir.parent / "eval_plain_kid750.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "images", "kid"])
        writer.writeheader()
        writer.writerows(result["results"])

    print(output)
    print(csv_path)
    print(json.dumps(result["results"][-1], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
