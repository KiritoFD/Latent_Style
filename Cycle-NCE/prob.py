import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import build_model_from_config
import json

def diagnose_checkpoint(config_path, ckpt_path):
    # 1. 加载配置和模型
    with open(config_path, "r") as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config["model"]).to(device)
    state = torch.load(ckpt_path, map_location=device)
    
    # 处理 torch.compile 产生的前缀
    sd = {k.replace("_orig_mod.", ""): v for k, v in state["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()

    # --- 维度 1: Embedding 几何分析 ---
    style_weights = model.style_emb.weight.detach() # [5, 160]
    # 计算余弦相似度
    weights_norm = F.normalize(style_weights, dim=1)
    sim_matrix = torch.mm(weights_norm, weights_norm.t()).cpu().numpy()
    
    print("\n[Analysis 1: Embedding Cosine Similarity]")
    print(sim_matrix)

    # --- 维度 2: 输出灵敏度探测 (Sensitivity Probe) ---
    # 构造一个随机的 Latent 输入
    dummy_content = torch.randn(1, 4, 32, 32).to(device)
    deltas = []
    
    with torch.no_grad():
        for i in range(config["model"]["num_styles"]):
            # 提取每一个风格对应的预测残差
            delta = model._predict_delta_from_context(dummy_content, 
                                                     style_code=style_weights[i:i+1], 
                                                     strength=1.0)
            deltas.append(delta)
    
    # 计算不同风格输出之间的平均 L1 差异
    all_deltas = torch.cat(deltas, dim=0) # [5, 4, 32, 32]
    diffs = []
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            diff = F.l1_loss(all_deltas[i], all_deltas[j]).item()
            diffs.append(diff)
    
    avg_diff = np.mean(diffs)
    print(f"\n[Analysis 2: Output Sensitivity (Mean ΔZ Diff)]")
    print(f"Average L1 difference between styles: {avg_diff:.6f}")

    # --- 可视化 ---
    plt.figure(figsize=(10, 4))
    
    # 相似度热力图
    plt.subplot(1, 2, 1)
    plt.imshow(sim_matrix, cmap='YlOrRd', vmin=0.8, vmax=1.0)
    plt.title("Style Similarity Heatmap")
    plt.colorbar()
    
    # SVD 降维观察分布 (160d -> 2d)
    U, S, V = torch.pca_lowrank(style_weights, q=2)
    coords = torch.matmul(style_weights, V[:, :2]).cpu().numpy()
    plt.subplot(1, 2, 2)
    plt.scatter(coords[:, 0], coords[:, 1], c=range(len(coords)), cmap='viridis', s=100)
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, f"Style {i}", fontsize=9)
    plt.title("Embedding PCA Projection (2D)")
    
    plt.tight_layout()
    plt.savefig("style_probe_report.png")
    print("\nReport saved to style_probe_report.png")

if __name__ == "__main__":
    diagnose_checkpoint("config.json", "../nce-swd_0.25-cl_0.01/epoch_0120.pt")