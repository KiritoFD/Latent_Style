import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.losses import RiemannianStyleLoss


def resolve_data_root(data_root: Path | None) -> Path:
    if data_root is not None:
        return data_root
    return Path(__file__).resolve().parents[1] / "sdxl-256"


def parse_styles(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def run_analysis(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_data_root(args.data_root)
    styles = parse_styles(args.styles)

    loss_fn = RiemannianStyleLoss(scale_factor=args.scale_factor).to(device)

    all_feats = []
    all_labels: list[str] = []

    print(f"Running Riemannian Log-Covariance Analysis on {device}...")
    print(f"Data root: {data_root}")

    with torch.no_grad():
        for style in styles:
            files = list((data_root / style).glob("*.pt"))[: args.max_per_style]
            if not files:
                continue

            feats_list = []
            for i in range(0, len(files), args.batch_size):
                batch_files = files[i : i + args.batch_size]
                x = torch.stack(
                    [torch.load(f, map_location=device, weights_only=True) for f in batch_files]
                )
                x_in = x * args.scale_factor
                x_lifted = loss_fn.spatial_shift_stack(x_in)
                log_cov = loss_fn.compute_log_covariance(x_lifted)

                b, c, _ = log_cov.shape
                idx = torch.triu_indices(c, c, device=log_cov.device)
                flat_feats = log_cov[:, idx[0], idx[1]]
                feats_list.append(flat_feats.cpu().numpy())

            if feats_list:
                all_feats.append(np.concatenate(feats_list))
                all_labels.extend([style] * len(files))
                print(f"Processed {style}: {len(files)} images")

    if not all_feats:
        print("No latents found for selected styles.")
        return

    X = np.concatenate(all_feats)
    X_norm = StandardScaler().fit_transform(X)

    score = silhouette_score(X_norm, all_labels)
    print(f"\nSilhouette Score: {score:.4f}")

    pca = PCA(n_components=2).fit_transform(X_norm)
    labels_np = np.array(all_labels)
    plt.figure(figsize=(10, 8))
    for style in styles:
        mask = labels_np == style
        if mask.any():
            plt.scatter(pca[mask, 0], pca[mask, 1], label=style, s=15, alpha=0.6)

    plt.title(f"Riemannian Log-Covariance Metric\nScore: {score:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(args.out)
    print(f"Saved {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--styles", type=str, default="photo,Hayao,vangogh,monet,cezanne")
    p.add_argument("--max-per-style", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--scale-factor", type=float, default=0.13025)
    p.add_argument("--out", type=Path, default=Path("riemannian_pca.png"))
    return p


if __name__ == "__main__":
    parser = build_parser()
    run_analysis(parser.parse_args())
