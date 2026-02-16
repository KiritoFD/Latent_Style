import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.spectral_norm as spectral_norm
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import random
from collections import Counter
import warnings
import sys

# 忽略不重要的警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 严苛配置 (Hard Thresholds)
# ==========================================
CONFIG = {
    "data_root": (Path(__file__).resolve().parent / "../../../sdxl-256").resolve(),
    "styles": ["photo", "Hayao", "monet", "cezanne", "vangogh"],
    "batch_size": 128,
    "epochs": 100,          # 给足时间收敛
    "lr": 1e-3,
    "seed": 42,
    "val_split": 0.2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    # --- 强制门槛 (达不到不许停) ---
    "target_min_recall": 0.85,  # 每一类 (包括凡高) 的 Recall 必须 > 60%
    "target_accuracy": 0.85     # 整体准确率必须 > 80%
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

# ==========================================
# 2. 模型: 允许学习统计特征 (Affine=True)
# ==========================================
class RobustStyleProbe(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        def res_block(in_c, out_c, stride=1):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)),
                nn.InstanceNorm2d(out_c, affine=True),
                nn.Mish(inplace=True),
                nn.Dropout2d(0.1),
            )

        self.layer1 = res_block(4, 32)
        self.layer2 = res_block(32, 64, stride=2)
        self.layer3 = res_block(64, 128)
        self.layer4 = res_block(128, 128, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = spectral_norm(nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat_vec = self.gap(x).flatten(1)
        return feat_vec, self.classifier(feat_vec)

# ==========================================
# 3. 数据加载: 仅依赖采样器平衡
# ==========================================
class LatentAugment:
    def __call__(self, x):
        # 几何增强，防止死记硬背像素
        if random.random() < 0.5: x = torch.flip(x, [2])
        if random.random() < 0.5: x = torch.rot90(x, 1, [1, 2])
        return x

def load_data_balanced():
    print(f"🚀 [Data Audit] Scanning {CONFIG['data_root']}...")
    all_lats, all_labels = [], []
    
    for idx, style in enumerate(CONFIG['styles']):
        style_dir = CONFIG['data_root'] / style
        files = list(style_dir.glob("*.pt"))
        count = len(files)
        print(f"   - {style:<10}: {count} samples")
        if count == 0:
            raise ValueError(f"❌ No data found for {style}!")

        for f in files:
            all_lats.append(torch.load(f))
            all_labels.append(idx)
            
    X = torch.stack(all_lats)
    Y = torch.tensor(all_labels)
    
    # 计算每个样本的采样权重 (Inverse Frequency)
    class_counts = Counter(all_labels)
    # weight = 1.0 / count
    weights_for_sampler = [1.0 / class_counts[y.item()] for y in Y]
    
    print("-" * 40)
    print("⚖️  Sampler Strategy: Force 1:1:1:1:1 distribution in Batches")
    
    return X, Y, torch.DoubleTensor(weights_for_sampler)

# ==========================================
# 4. 训练循环: 达标才保存
# ==========================================
def run_strict_training():
    # 1. 准备数据
    X, Y, sampler_weights = load_data_balanced()
    dataset = TensorDataset(X, Y)
    
    # 划分验证集
    indices = list(range(len(dataset)))
    split_idx = int((1 - CONFIG['val_split']) * len(dataset))
    random.shuffle(indices) # 随机打乱
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # 训练集：使用 WeightedRandomSampler (这是关键!)
    train_weights = sampler_weights[train_indices]
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_indices), # 保持 Epoch 大小一致
        replacement=True                # 允许重复采样 (小样本类会被多次采样)
    )
    
    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False) # 验证集看真实分布
    
    # 2. 模型与优化器
    model = RobustStyleProbe(len(CONFIG['styles'])).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
    criterion = nn.CrossEntropyLoss() # 普通 Loss，不加权
    augmentor = LatentAugment()

    print(f"\n🔥 Starting Training... Goals: Min Recall > {CONFIG['target_min_recall']}, Acc > {CONFIG['target_accuracy']}")
    
    best_min_recall = 0.0
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(CONFIG['device']), by.to(CONFIG['device'])
            bx = torch.stack([augmentor(x) for x in bx]) # 实时增强
            
            optimizer.zero_grad()
            _, logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # --- 严苛验证环节 ---
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(CONFIG['device']), by.to(CONFIG['device'])
                _, logits = model(bx)
                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(by.cpu().numpy())
        
        # 计算每个类的 Recall
        report = classification_report(all_targets, all_preds, target_names=CONFIG['styles'], output_dict=True, zero_division=0)
        
        # 提取关键指标
        overall_acc = report['accuracy']
        recalls = [report[s]['recall'] for s in CONFIG['styles']]
        min_class_recall = min(recalls) # 寻找短板 (木桶效应)
        
        # 学习率调整 (基于宏平均F1)
        macro_f1 = report['macro avg']['f1-score']
        scheduler.step(macro_f1)
        
        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {overall_acc:.2%} | Min Recall: {min_class_recall:.2%}")
            # 打印详细 Recall 分布
            recall_str = ", ".join([f"{s[:3]}:{r:.2f}" for s, r in zip(CONFIG['styles'], recalls)])
            print(f"      [Recalls] {recall_str}")

        # --- 核心门槛判定 ---
        # 只有当【所有类 Recall > 门槛】且【整体 Acc > 门槛】时，才视为成功并保存
        if min_class_recall > best_min_recall:
            best_min_recall = min_class_recall
            # 暂存为最佳模型 (不管是否达标，至少存个最好的)
            torch.save(model.state_dict(), str((Path(__file__).resolve().parent / "robust_style_probe_best.pth")))
        
        if min_class_recall >= CONFIG['target_min_recall'] and overall_acc >= CONFIG['target_accuracy']:
            print("\n" + "="*60)
            print(f"🎉 SUCCESS: Thresholds Met at Epoch {epoch+1}!")
            print(f"   Accuracy: {overall_acc:.2%} (Target: {CONFIG['target_accuracy']})")
            print(f"   Min Recall: {min_class_recall:.2%} (Target: {CONFIG['target_min_recall']})")
            print("="*60)
            torch.save(model.state_dict(), str((Path(__file__).resolve().parent / "robust_style_probe_final.pth")))
            break # 达到目标，提前结束

    print(f"\n✅ Training Ended. Best Min Recall achieved: {best_min_recall:.2%}")
    if best_min_recall < CONFIG['target_min_recall']:
        print(f"⚠️ Warning: Target recall ({CONFIG['target_min_recall']}) NOT met. Try increasing epochs or adjusting architecture.")
    else:
        print("💾 Valid model saved to 'robust_style_probe_final.pth'")

    # ==========================================
    # 5. 最终审计 (可视化验证)
    # ==========================================
    print("\n🔬 Final Audit...")
    # 加载最好的那个
    load_path = (Path(__file__).resolve().parent / "robust_style_probe_final.pth") if (Path(__file__).resolve().parent / "robust_style_probe_final.pth").exists() else (Path(__file__).resolve().parent / "robust_style_probe_best.pth")
    model.load_state_dict(torch.load(load_path))
    model.eval()
    
    all_feats, all_targets = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(CONFIG['device']), by.to(CONFIG['device'])
            f, _ = model(bx)
            all_feats.extend(f.cpu().numpy())
            all_targets.extend(by.cpu().numpy())
            
    sil_score = silhouette_score(np.array(all_feats), np.array(all_targets))
    print(f"👉 Silhouette Score: {sil_score:.4f}")
    
    if sil_score > 0.2:
        print("✅ PASS: Feature space is disentangled.")
    else:
        print("❌ FAIL: Feature space is still entangled.")

if __name__ == "__main__":
    run_strict_training()