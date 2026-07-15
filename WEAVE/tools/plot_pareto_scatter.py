"""绘制 WD-VF vs all baselines 散点图: 横轴 1-LPIPS (内容保真度↑), 纵轴 CLIP-S (风格强度↑).
SaMam 画 81 checkpoint 收敛曲线; 我们只画 3 个代表性点."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import csv
import re
import os

# 字体设置 (中文)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== Baseline 数据 (统一评估协议: HF CLIP ViT-B/32, LPIPS-Alex, 512px, distinct5) =====
baselines = {
    # 方法名: (CLIP-S, LPIPS, 类别, 标记大小)
    "Identity":      (0.6933, 0.0000, "baseline", 60),
    "AdaIN":         (0.6679, 0.7425, "classical", 60),
    "WCT (VGG19)":   (0.7063, 0.6348, "classical", 60),
    "SDEdit s=0.10": (0.7188, 0.3183, "diffusion", 45),
    "SDEdit s=0.20": (0.7340, 0.3492, "diffusion", 45),
    "SDEdit s=0.35": (0.7797, 0.4508, "diffusion", 55),
    "SDEdit s=0.40": (0.7934, 0.4826, "diffusion", 55),
    "SD-Turbo":      (0.6933, 0.0033, "diffusion", 60),
    "StyleID":       (0.8223, 0.5523, "diffusion", 60),
    "CUT":           (0.7137, 0.3743, "gan", 60),
    "SaMST":         (0.6183, 0.7490, "mamba", 60),
    "SeeDream":      (0.7198, 0.4767, "diffusion", 60),
}

# ===== SaMam 81 checkpoint 收敛曲线 (从 CSV 读取) =====
samam_curve = []  # [(step, clip_s, lpips), ...]
csv_path = "tools/samam_distinct5_scratch/curve_metrics_hf.csv"
if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_dir = row.get('image_dir', '')
            m = re.search(r'step_(\d+)', img_dir)
            step = int(m.group(1)) if m else 0
            clip_s = float(row['clip_style'])
            lpips = float(row['content_lpips'])
            samam_curve.append((step, clip_s, lpips))
    samam_curve.sort(key=lambda x: x[0])
    print(f"Loaded SaMam curve: {len(samam_curve)} checkpoints, "
          f"step {samam_curve[0][0]}-{samam_curve[-1][0]}")

# ===== WD-VF (Ours) 实验点 — 只保留 3 个代表性点 =====
ours = {
    # 实验名: (CLIP-S, LPIPS, 颜色/标记)
    "WD-VF (SOTA)":      (0.7213, 0.2868, "#FF4444", "*", 220),
    "WD-VF (4-step)":    (0.7166, 0.2811, "#FF8888", "D", 90),
    "WD-VF (low-LPIPS)": (0.7150, 0.2597, "#FFAAAA", "s", 70),
}

# 类别颜色映射
cat_colors = {
    "baseline": "#888888",
    "classical": "#2196F3",
    "diffusion": "#FF9800",
    "gan": "#4CAF50",
    "mamba": "#9C27B0",
}

# ===== 绘图 =====
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# 1. 绘制 SaMam 收敛曲线 (81 checkpoint 连线)
if samam_curve:
    sx = [1.0 - p[2] for p in samam_curve]  # 1-LPIPS
    sy = [p[1] for p in samam_curve]         # CLIP-S
    # 画连线
    ax.plot(sx, sy, color='#9C27B0', alpha=0.4, linewidth=1.5, zorder=2,
            linestyle='-', marker='.', markersize=4)
    # 标注起点和终点
    ax.scatter([sx[0]], [sy[0]], s=50, c='#9C27B0', marker='o',
               edgecolors='black', linewidths=0.5, zorder=4)
    ax.annotate("SaMam step250\n(CLIP-S=0.52)", (sx[0], sy[0]),
                xytext=(sx[0]-0.08, sy[0]-0.03), fontsize=7, color='#9C27B0',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#9C27B0', alpha=0.5))
    ax.scatter([sx[-1]], [sy[-1]], s=80, c='#9C27B0', marker='s',
               edgecolors='black', linewidths=0.8, zorder=4)
    ax.annotate("SaMam step20K\n(CLIP-S=0.5816\n<Identity 0.6933)",
                (sx[-1], sy[-1]), xytext=(sx[-1]+0.01, sy[-1]-0.05),
                fontsize=7.5, color='#9C27B0', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#F3E5F5',
                          edgecolor='#9C27B0', alpha=0.8))

# 2. 绘制其他 baselines
for name, (clip, lpips, cat, sz) in baselines.items():
    x = 1.0 - lpips  # 内容保真度
    y = clip
    color = cat_colors[cat]
    ax.scatter(x, y, s=sz, c=color, alpha=0.8, edgecolors='black',
               linewidths=0.5, zorder=3)
    # 标签偏移
    offset_x, offset_y = 0.005, 0.003
    if name == "Identity":
        offset_x, offset_y = 0.005, 0.008
    elif name == "SD-Turbo":
        offset_x, offset_y = 0.005, 0.008
    elif name == "SDEdit s=0.40":
        offset_x, offset_y = -0.03, 0.006
    elif name == "SDEdit s=0.35":
        offset_x, offset_y = 0.005, 0.006
    elif name == "StyleID":
        offset_x, offset_y = -0.02, 0.008
    ax.annotate(name, (x, y), xytext=(x + offset_x, y + offset_y),
                fontsize=8, color=color, fontweight='bold')

# 3. 绘制 WD-VF 实验点 (3 个代表性点)
for name, (clip, lpips, color, marker, sz) in ours.items():
    x = 1.0 - lpips
    y = clip
    is_sota = "SOTA" in name
    lw = 2.5 if is_sota else 1.5
    ax.scatter(x, y, s=sz, c=color, marker=marker, edgecolors='darkred',
               linewidths=lw, zorder=6)
    # 标签
    if is_sota:
        ax.annotate(name, (x, y), xytext=(x-0.06, y+0.015),
                    fontsize=11, color="#FF0000", fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEEEE',
                              edgecolor='red', alpha=0.85))
    else:
        ax.annotate(name, (x, y), xytext=(x+0.005, y+0.005),
                    fontsize=8, color="#CC4444", fontstyle='italic')

# 4. 标注 WD-VF Pareto 区域高亮
t11_x, t11_y = 1.0 - 0.2868, 0.7213
rect = plt.Rectangle((0.69, 0.71), 0.06, 0.018, fill=True,
                      facecolor='#FF000008', edgecolor='red',
                      linestyle='--', linewidth=1.5, zorder=1)
ax.add_patch(rect)

# 5. 图例 (按类别)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=cat_colors["baseline"],
           markersize=8, markeredgecolor='black', label='Baseline'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=cat_colors["classical"],
           markersize=8, markeredgecolor='black', label='Classical (inf)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=cat_colors["diffusion"],
           markersize=8, markeredgecolor='black', label='Diffusion'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=cat_colors["gan"],
           markersize=8, markeredgecolor='black', label='GAN'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=cat_colors["mamba"],
           markersize=8, markeredgecolor='black', label='Mamba'),
    Line2D([0], [0], color='#9C27B0', alpha=0.5, linewidth=1.5,
           marker='.', markersize=5, label='SaMam 收敛曲线 (81 ckpt)'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#FF4444',
           markersize=14, markeredgecolor='darkred', label='WD-VF (Ours)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

# 6. 轴标签和样式
ax.set_xlabel("1 - LPIPS (内容保真度 Content Fidelity ↑)", fontsize=12, fontweight='bold')
ax.set_ylabel("CLIP-S (风格转移强度 Style Transfer ↑)", fontsize=12, fontweight='bold')
ax.set_title("WD-VF vs All Baselines: Style-Content Trade-off\n"
             "(HF CLIP ViT-B/32, LPIPS-Alex, 512px, distinct5, 750 pairs)",
             fontsize=13, fontweight='bold')

# 坐标轴范围: 排除 Identity/SD-Turbo 的极右端 (x≈1.0), 让 WD-VF 处于右上方
ax.set_xlim(0.20, 0.82)
ax.set_ylim(0.50, 0.85)
ax.grid(True, alpha=0.3, linestyle='-')

# Identity baseline 水平线: 下方为"伪风格转移"区域
ax.axhline(y=0.6933, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)
ax.annotate("Identity baseline (CLIP-S=0.6933)\n↓ 下方: 伪风格转移 (CLIP-S < Identity)",
            xy=(0.50, 0.6933), xytext=(0.22, 0.555),
            fontsize=7.5, color='gray', fontweight='bold', alpha=0.8,
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

# Pareto 最优方向标注
ax.annotate("Pareto最优\n(高风格+高保真)", xy=(0.78, 0.83),
            fontsize=9, color='green', fontweight='bold', alpha=0.6,
            ha='center')

plt.tight_layout()
os.makedirs("docs/72", exist_ok=True)
os.makedirs("aaai2027_v2", exist_ok=True)
plt.savefig("docs/72/pareto_scatter_all_baselines.png", dpi=200, bbox_inches='tight')
plt.savefig("docs/72/pareto_scatter_all_baselines.pdf", bbox_inches='tight')
plt.savefig("aaai2027_v2/fig_all_baselines_scatter.png", dpi=200, bbox_inches='tight')
plt.savefig("aaai2027_v2/fig_all_baselines_scatter.pdf", bbox_inches='tight')
print("Saved: docs/72/ + aaai2027_v2/fig_all_baselines_scatter.png/pdf")
