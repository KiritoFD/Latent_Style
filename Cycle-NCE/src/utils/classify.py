import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.spectral_norm as spectral_norm
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, silhouette_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


CONFIG = {
    "data_root": (Path(__file__).resolve().parent / "../../../sdxl-256").resolve(),
    "styles": ["photo", "Hayao", "monet", "cezanne", "vangogh"],
    "batch_size": 128,
    "epochs": 100,
    "lr": 1e-3,
    "seed": 42,
    "val_split": 0.2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "target_min_recall": 0.85,
    "target_accuracy": 0.85,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])


class RobustStyleProbe(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()

        def res_block(in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)),
                nn.InstanceNorm2d(out_c, affine=True),
                nn.Mish(inplace=True),
                nn.Dropout2d(0.1),
            )

        # 7-block mirror: 3 stem + 4 backend
        self.stem = nn.Sequential(
            res_block(4, 16, stride=1),
            res_block(16, 32, stride=1),
            res_block(32, 32, stride=1),
        )
        self.layer1 = res_block(32, 32)
        self.layer2 = res_block(32, 64, stride=2)
        self.layer3 = res_block(64, 128)
        self.layer4 = res_block(128, 128, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = spectral_norm(nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat_vec = self.gap(x).flatten(1)
        return feat_vec, self.classifier(feat_vec)


class LatentAugment:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            x = torch.flip(x, [2])
        if random.random() < 0.5:
            x = torch.rot90(x, 1, [1, 2])
        return x


def _load_latent(path: Path) -> torch.Tensor:
    x = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(x, dict) and "latent" in x:
        x = x["latent"]
    if x.ndim == 4 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim != 3:
        raise ValueError(f"Unexpected latent shape {tuple(x.shape)} in {path}")
    return x.float().contiguous()


def load_data_balanced():
    print(f"[Data Audit] Scanning {CONFIG['data_root']}...")
    all_lats, all_labels = [], []

    for idx, style in enumerate(CONFIG["styles"]):
        style_dir = CONFIG["data_root"] / style
        files = sorted(style_dir.glob("*.pt"))
        count = len(files)
        print(f"  - {style:<10}: {count} samples")
        if count == 0:
            raise ValueError(f"No data found for {style}! dir={style_dir}")

        for f in files:
            all_lats.append(_load_latent(f))
            all_labels.append(idx)

    X = torch.stack(all_lats)
    Y = torch.tensor(all_labels)

    class_counts = Counter(all_labels)
    weights_for_sampler = [1.0 / class_counts[y.item()] for y in Y]

    print("-" * 40)
    print("Sampler Strategy: Force 1:1:1:1:1 distribution in batches")

    return X, Y, torch.DoubleTensor(weights_for_sampler)


def run_strict_training() -> None:
    X, Y, sampler_weights = load_data_balanced()
    dataset = TensorDataset(X, Y)

    indices = list(range(len(dataset)))
    split_idx = int((1 - CONFIG["val_split"]) * len(dataset))
    random.shuffle(indices)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_weights = sampler_weights[train_indices]
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_indices),
        replacement=True,
    )

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    script_dir = Path(__file__).resolve().parent

    def run_audit_and_pca(loaded_model: RobustStyleProbe) -> None:
        loaded_model.eval()
        all_feats, all_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(CONFIG["device"]), by.to(CONFIG["device"])
                f, _ = loaded_model(bx)
                all_feats.extend(f.cpu().numpy())
                all_targets.extend(by.cpu().numpy())

        feats = np.array(all_feats)
        targets = np.array(all_targets)
        sil_score = silhouette_score(feats, targets)
        print(f"Silhouette Score: {sil_score:.4f}")
        if sil_score > 0.2:
            print("PASS: Feature space is disentangled.")
        else:
            print("FAIL: Feature space is still entangled.")

        pca = PCA(n_components=2, random_state=CONFIG["seed"])
        emb = pca.fit_transform(feats)
        plt.figure(figsize=(9, 7))
        for sid, style_name in enumerate(CONFIG["styles"]):
            mask = targets == sid
            if np.any(mask):
                plt.scatter(emb[mask, 0], emb[mask, 1], s=12, alpha=0.55, label=style_name)
        plt.title(f"Latent Probe PCA | silhouette={sil_score:.4f}")
        plt.legend()
        plt.tight_layout()
        out_png = script_dir / "latent_probe_pca.png"
        plt.savefig(out_png, dpi=220)
        plt.close()
        print(f"Saved PCA plot: {out_png}")

    final_ckpt = script_dir / "robust_style_probe_final.pth"
    best_ckpt = script_dir / "robust_style_probe_best.pth"
    ckpt_path = final_ckpt if final_ckpt.exists() else (best_ckpt if best_ckpt.exists() else None)
    if ckpt_path is not None:
        print(f"Found checkpoint: {ckpt_path}")
        model = RobustStyleProbe(len(CONFIG["styles"])).to(CONFIG["device"])
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict, strict=True)
        print("Skip training and run PCA-only audit.")
        run_audit_and_pca(model)
        return

    model = RobustStyleProbe(len(CONFIG["styles"])).to(CONFIG["device"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    criterion = nn.CrossEntropyLoss()
    augmentor = LatentAugment()

    print(
        f"\nStarting Training... Goals: Min Recall > {CONFIG['target_min_recall']}, "
        f"Acc > {CONFIG['target_accuracy']}"
    )

    best_min_recall = 0.0
    best_state_dict = None
    best_epoch = 0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(CONFIG["device"]), by.to(CONFIG["device"])
            bx = torch.stack([augmentor(x) for x in bx])

            optimizer.zero_grad()
            _, logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(CONFIG["device"]), by.to(CONFIG["device"])
                _, logits = model(bx)
                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(by.cpu().numpy())

        report = classification_report(
            all_targets,
            all_preds,
            target_names=CONFIG["styles"],
            output_dict=True,
            zero_division=0,
        )

        overall_acc = float(report["accuracy"])
        recalls = [float(report[s]["recall"]) for s in CONFIG["styles"]]
        min_class_recall = float(min(recalls))

        macro_f1 = float(report["macro avg"]["f1-score"])
        scheduler.step(macro_f1)

        if (epoch + 1) % 5 == 0:
            print(
                f"  Epoch {epoch+1:03d} | Loss: {total_loss/max(len(train_loader),1):.4f} "
                f"| Acc: {overall_acc:.2%} | Min Recall: {min_class_recall:.2%}"
            )
            recall_str = ", ".join([f"{s[:3]}:{r:.2f}" for s, r in zip(CONFIG["styles"], recalls)])
            print(f"      [Recalls] {recall_str}")

        if min_class_recall > best_min_recall:
            best_min_recall = min_class_recall
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "meta": {
                        "arch": "RobustStyleProbe_mirror7",
                        "styles": list(CONFIG["styles"]),
                        "best_min_recall": float(best_min_recall),
                        "best_epoch": int(best_epoch),
                    },
                },
                str(script_dir / "robust_style_probe_best.pth"),
            )

        if min_class_recall >= CONFIG["target_min_recall"] and overall_acc >= CONFIG["target_accuracy"]:
            print("\n" + "=" * 60)
            print(f"SUCCESS: Thresholds met at epoch {epoch+1}")
            print(f"  Accuracy: {overall_acc:.2%} (Target: {CONFIG['target_accuracy']})")
            print(f"  Min Recall: {min_class_recall:.2%} (Target: {CONFIG['target_min_recall']})")
            print("=" * 60)
            cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": cur_state,
                    "meta": {
                        "arch": "RobustStyleProbe_mirror7",
                        "styles": list(CONFIG["styles"]),
                        "best_min_recall": float(best_min_recall),
                        "best_epoch": int(best_epoch),
                        "hit_threshold": True,
                    },
                },
                str(script_dir / "robust_style_probe_final.pth"),
            )
            break

    print(f"\nTraining Ended. Best Min Recall achieved: {best_min_recall:.2%}")
    if best_min_recall < CONFIG["target_min_recall"]:
        print(f"Warning: Target recall ({CONFIG['target_min_recall']}) NOT met.")
    else:
        print("Valid model saved to robust_style_probe_final.pth")

    print("\nFinal Audit...")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict, strict=True)
    else:
        load_path = (
            script_dir / "robust_style_probe_final.pth"
            if (script_dir / "robust_style_probe_final.pth").exists()
            else script_dir / "robust_style_probe_best.pth"
        )
        payload = torch.load(load_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict, strict=True)

    run_audit_and_pca(model)


if __name__ == "__main__":
    run_strict_training()
