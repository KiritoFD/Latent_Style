import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

from classify import RobustStyleProbe


def _load_latent_4d(path: Path, device: torch.device) -> torch.Tensor:
    x = torch.load(path, map_location=device)
    if isinstance(x, dict) and "latent" in x:
        x = x["latent"]
    if x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Unexpected latent shape {tuple(x.shape)} in {path}")
    return x.float()


def visualize() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = Path(__file__).resolve().parent
    data_root = (script_dir / "../../../sdxl-256").resolve()
    ckpt_path = (script_dir / "robust_style_probe_final.pth").resolve()
    styles = ["photo", "Hayao", "monet", "cezanne", "vangogh"]

    model = RobustStyleProbe(num_classes=len(styles)).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    all_feats = []
    all_logits = []
    all_labels = []

    print(f"Extracting features on {device}...")
    with torch.no_grad():
        for i, style_name in enumerate(styles):
            files = sorted((data_root / style_name).glob("*.pt"))
            for j in range(0, len(files), 64):
                batch_files = files[j : j + 64]
                batch = torch.cat([_load_latent_4d(f, device=device) for f in batch_files], dim=0)
                feat, logits = model(batch)
                all_feats.append(feat.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                all_labels.extend([i] * len(batch_files))

    feats = np.concatenate(all_feats, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    labels = np.array(all_labels)
    preds = np.argmax(logits, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(feats)
    for i, style_name in enumerate(styles):
        mask = labels == i
        axes[0].scatter(feats_2d[mask, 0], feats_2d[mask, 1], label=style_name, alpha=0.6, s=15)
    axes[0].set_title("Style Manifold (PCA)")
    axes[0].legend()

    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=styles, yticklabels=styles, ax=axes[1])
    axes[1].set_title("Confusion Matrix")

    print(classification_report(labels, preds, target_names=styles, digits=4, zero_division=0))

    plt.tight_layout()
    plt.savefig("probe_analysis.png", dpi=300)
    print("Results saved to probe_analysis.png")


if __name__ == "__main__":
    visualize()
