import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm

# ==========================================
# 1. 定义探针网络 (Style Probe Architecture)
# ==========================================
class StyleProbe(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 这是一个专门针对 VAE Latent (4通道) 设计的轻量级特征提取器
        self.features = nn.Sequential(
            # [Layer 1] 核心去内容层
            # InstanceNorm 强行抹除均值和方差，只留纹理
            nn.InstanceNorm2d(4, affine=False), 
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [Layer 2] 下采样 + 特征聚合
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16x16
            nn.InstanceNorm2d(128, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [Layer 3] 再次下采样
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 8x8
            nn.InstanceNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [Global Pooling] 空间维度坍缩，得到纯风格向量
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten()
        )
        # 分类头 (训练完后可以丢弃，我们只需要 features)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return feat, logits

# ==========================================
# 2. 数据准备与训练流程
# ==========================================
def train_and_verify():
    # 配置
    data_root = Path("sdxl-256")  # 你的数据路径
    styles = ["photo", "Hayao", "monet", "cezanne", "vangogh"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 50
    lr = 1e-3
    save_path = "sdxl_style_probe.pth"

    print(f"🚀 初始化训练... 设备: {device}")

    # 加载数据 (全部加载到 VRAM，因为数据量很小，速度最快)
    all_lats, all_labels = [], []
    for i, s in enumerate(styles):
        files = list((data_root/s).glob("*.pt"))
        print(f"  Loading {s}: {len(files)} images")
        for f in files:
            all_lats.append(torch.load(f))
            all_labels.append(i)
    
    X = torch.stack(all_lats).to(device)
    Y = torch.tensor(all_labels).to(device)
    
    # 划分 80% 训练, 20% 验证
    dataset = TensorDataset(X, Y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 初始化模型
    model = StyleProbe(num_classes=len(styles)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- 训练循环 ---
    print("\n🔥 开始训练探针...")
    history = {'train_loss': [], 'val_acc': []}
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            _, logits = model(bx) # 只用 logits 算 Loss
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                _, logits = model(bx)
                pred = logits.argmax(dim=1)
                correct += (pred == by).sum().item()
                total += by.size(0)
        
        acc = correct / total
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_acc'].append(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path) # 保存最佳模型
            
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.2%}")

    print(f"\n✅ 训练完成! 最佳验证集准确率: {best_acc:.2%}")
    
    # ==========================================
    # 3. 终极可用性验证 (Visualization & Silhouette)
    # ==========================================
    print("\n🔍 执行终极特征验证...")
    
    # 加载最佳权重
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    # 提取所有验证集特征
    feats = []
    labels = []
    with torch.no_grad():
        for bx, by in val_loader:
            f, _ = model(bx)
            feats.append(f.cpu().numpy())
            labels.append(by.cpu().numpy())
            
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    
    # 1. 计算 Silhouette Score
    sil_score = silhouette_score(feats, labels)
    print(f"👉 最终特征轮廓系数 (Silhouette Score): {sil_score:.4f}")
    
    if sil_score > 0.2:
        print("🎉 [判定结果]: 风格特征高度解耦，该探针绝对可用！")
    elif sil_score > 0:
        print("⚠️ [判定结果]: 风格特征基本可用，但建议增加训练轮数或数据量。")
    else:
        print("❌ [判定结果]: 探针训练失败，请检查数据。")

    # 2. 绘制 PCA 可视化
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(feats)
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("hls", len(styles))
    for i, s in enumerate(styles):
        idx = labels == i
        plt.scatter(feats_2d[idx, 0], feats_2d[idx, 1], label=s, color=colors[i], s=20, alpha=0.7)
    
    plt.title(f"Style Probe Feature Space\nAccuracy: {best_acc:.2%} | Silhouette: {sil_score:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("probe_verification_result.png")
    print("📊 可视化结果已保存至 'probe_verification_result.png'")

if __name__ == "__main__":
    train_and_verify()