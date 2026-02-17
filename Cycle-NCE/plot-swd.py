import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.losses import LatentStyleLoss


def parse_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def extract_swd_features(loss_module, x, spatial_samples=64):
    with torch.no_grad():
        feat = loss_module.lifter(x * loss_module.scale_factor)
        b, c, h, w = feat.shape
        f_flat = feat.view(b, c, -1).permute(0, 2, 1)
        projected = f_flat @ loss_module.projections
        sorted_proj, _ = torch.sort(projected, dim=1)
        idx = torch.linspace(0, h * w - 1, spatial_samples, device=x.device).long()
        return sorted_proj[:, idx, :].reshape(b, -1)


def load_style_latents(data_root, styles, max_per_style, device):
    style_data = {}
    counts = {}
    for style in styles:
        files = list((data_root / style).glob("*.pt"))[:max_per_style]
        if not files:
            continue
        tensor = torch.stack([torch.load(f, map_location=device, weights_only=True) for f in files])
        style_data[style] = tensor
        counts[style] = len(files)
    return style_data, counts


def run_single_config(style_data, styles, device, lift_dim, rff_sigma, num_projections, spatial_samples, batch_size):
    loss_module = LatentStyleLoss(
        in_channels=4,
        lift_dim=lift_dim,
        rff_sigma=rff_sigma,
        num_projections=num_projections,
    ).to(device)

    all_feats = []
    all_labels = []
    for style in styles:
        if style not in style_data:
            continue
        x = style_data[style]
        chunks = []
        for i in range(0, len(x), batch_size):
            chunks.append(extract_swd_features(loss_module, x[i : i + batch_size], spatial_samples=spatial_samples).cpu().numpy())
        all_feats.append(np.concatenate(chunks))
        all_labels.extend([style] * len(x))

    X = np.concatenate(all_feats)
    X_norm = StandardScaler().fit_transform(X)
    score = float(silhouette_score(X_norm, all_labels))
    return score, X_norm, np.array(all_labels)


def plot_best_projection(X_norm, labels, styles, score, lift_dim, rff_sigma, out_path):
    pca = PCA(n_components=2).fit_transform(X_norm)
    plt.figure(figsize=(10, 7))
    for style in styles:
        mask = labels == style
        if mask.any():
            plt.scatter(pca[mask, 0], pca[mask, 1], label=style, s=20, alpha=0.6)
    plt.title(
        "Latent RFF-SWD Best Config\n"
        f"sil={score:.4f}, lift_dim={lift_dim}, sigma={rff_sigma}"
    )
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_path)


def write_results_csv(rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lift_dim", "rff_sigma", "num_projections", "spatial_samples", "silhouette"])
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Scan RFF-SWD hyperparameters.")
    p.add_argument("--data-root", type=Path, default=Path("../sdxl-256"))
    p.add_argument("--styles", type=str, default="photo,Hayao,vangogh,monet,cezanne")
    p.add_argument("--max-per-style", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sigmas", type=str, default="1.0,2.0,3.0,5.0")
    p.add_argument("--lift-dims", type=str, default="256,512,1024")
    p.add_argument("--num-projections", type=int, default=128)
    p.add_argument("--spatial-samples", type=int, default=64)
    p.add_argument("--out-csv", type=Path, default=Path("swd_scan_results.csv"))
    p.add_argument("--out-plot", type=Path, default=Path("swd_separability_analysis.png"))
    return p


def run_scan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    styles = parse_list(args.styles, str)
    sigmas = parse_list(args.sigmas, float)
    lift_dims = parse_list(args.lift_dims, int)

    print(f"Device: {device}")
    print(f"Loading latents from: {args.data_root}")
    style_data, counts = load_style_latents(args.data_root, styles, args.max_per_style, device)
    if not style_data:
        print("No latent files found; nothing to analyze.")
        return

    valid_styles = [s for s in styles if s in style_data]
    print("Loaded styles:", ", ".join(f"{s}({counts[s]})" for s in valid_styles))

    results = []
    best = None
    for lift_dim in lift_dims:
        for sigma in sigmas:
            score, X_norm, labels = run_single_config(
                style_data=style_data,
                styles=valid_styles,
                device=device,
                lift_dim=lift_dim,
                rff_sigma=sigma,
                num_projections=args.num_projections,
                spatial_samples=args.spatial_samples,
                batch_size=args.batch_size,
            )
            row = {
                "lift_dim": lift_dim,
                "rff_sigma": sigma,
                "num_projections": args.num_projections,
                "spatial_samples": args.spatial_samples,
                "silhouette": score,
            }
            results.append(row)
            print(f"lift_dim={lift_dim:4d}, sigma={sigma:>4.1f} -> silhouette={score:.4f}")

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "lift_dim": lift_dim,
                    "sigma": sigma,
                    "X_norm": X_norm,
                    "labels": labels,
                }

    write_results_csv(results, args.out_csv)
    print(f"Saved scan table: {args.out_csv}")

    assert best is not None
    plot_best_projection(
        X_norm=best["X_norm"],
        labels=best["labels"],
        styles=valid_styles,
        score=best["score"],
        lift_dim=best["lift_dim"],
        rff_sigma=best["sigma"],
        out_path=args.out_plot,
    )
    print(
        "Best config: "
        f"lift_dim={best['lift_dim']}, sigma={best['sigma']}, silhouette={best['score']:.4f}"
    )
    print(f"Saved best PCA plot: {args.out_plot}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()
