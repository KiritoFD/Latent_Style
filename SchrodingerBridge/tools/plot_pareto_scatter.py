"""绘制 FC-SB vs all baselines 散点图: 横轴 1-LPIPS (内容保真度↑), 纵轴 CLIP-S (风格强度↑)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

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
    "SaMam (20K)":   (0.5816, 0.2434, "mamba", 70),
    "SeeDream":      (0.7198, 0.4767, "diffusion", 60),
}

# ===== FC-SB (Ours) 实验点 =====
ours = {
    # 实验名: (CLIP-S, LPIPS, 颜色/标记)
    "T11 (SOTA)":    (0.7213, 0.2868, "#FF4444", "*", 200),
    "T22 (Plan D)":  (0.7150, 0.2597, "#FF8888", "D", 70),
    "R1 (depth=2)":  (0.7172, 0.2705, "#FFAAAA", "s", 50),
    "R2 (dim=32)":   (0.7153, 0.2705, "#FFAAAA", "s", 50),
    "R3 (gate=0)":   (0.7080, 0.2641, "#FFAAAA", "s", 50),
    "T21 (Plan C)":  (0.7174, 0.2790, "#FF8888", "D", 70),
    "T20 (Plan B)":  (0.7037, 0.2722, "#FF8888", "D", 70),
    "T25 (Plan G)":  (0.7205, 0.3069, "#FFCCCC", "^", 55),
    "T26 (Plan H)":  (0.7200, 0.3085, "#FFCCCC", "^", 55),
    "T23 (Plan E)":  (0.7215, 0.3856, "#FFDDDD", "v", 50),
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

# 1. 绘制 baselines
for name, (clip, lpips, cat, sz) in baselines.items():
    x = 1.0 - lpips  # 内容保真度
    y = clip
    color = cat_colors[cat]
    ax.scatter(x, y, s=sz, c=color, alpha=0.8, edgecolors='black', linewidths=0.5, zorder=3)
    # 标签偏移
    offset_x, offset_y = 0.005, 0.003
    if name == "Identity":
        offset_x, offset_y = 0.005, 0.008
    elif name == "SD-Turbo":
        offset_x, offset_y = 0.005, 0.008
    elif name == "SaMam (20K)":
        offset_x, offset_y = 0.008, -0.006
    elif name == "SDEdit s=0.40":
        offset_x, offset_y = -0.03, 0.006
    elif name == "SDEdit s=0.35":
        offset_x, offset_y = 0.005, 0.006
    elif name == "StyleID":
        offset_x, offset_y = -0.02, 0.008
    ax.annotate(name, (x, y), xytext=(x + offset_x, y + offset_y),
                fontsize=8, color=color, fontweight='bold')

# 2. 绘制 FC-SB 实验点
for name, (clip, lpips, color, marker, sz) in ours.items():
    x = 1.0 - lpips
    y = clip
    is_sota = "SOTA" in name
    lw = 2.0 if is_sota else 1.0
    ax.scatter(x, y, s=sz, c=color, marker=marker, edgecolors='darkred',
               linewidths=lw, zorder=5, label=name if is_sota else "")
    # 标签
    offset_x, offset_y = 0.003, 0.004
    if "SOTA" in name:
        offset_x, offset_y = 0.005, -0.01
        ax.annotate(name, (x, y), xytext=(x + offset_x, y + offset_y),
                    fontsize=10, color="#FF0000", fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEEEE', edgecolor='red', alpha=0.7))
    else:
        ax.annotate(name, (x, y), xytext=(x + offset_x, y + offset_y),
                    fontsize=7, color="#CC4444", fontstyle='italic')

# 3. 绘制 T11 Pareto 区域高亮
# T11 是 FC-SB 最优点, 画一个虚线框标注
t11_x, t11_y = 1.0 - 0.2868, 0.7213
rect = plt.Rectangle((0.69, 0.71), 0.045, 0.018, fill=True, facecolor='#FF000008',
                      edgecolor='red', linestyle='--', linewidth=1.5, zorder=1)
ax.add_patch(rect)

# 4. 标注1:8 trade-off 线
# 从 T11 到各实验, 线性拟合
tradeoff_pts = [(1-0.2868, 0.7213), (1-0.2597, 0.7150), (1-0.2722, 0.7037),
                (1-0.2705, 0.7172), (1-0.2641, 0.7080), (1-0.2790, 0.7174)]
xs = [p[0] for p in tradeoff_pts]
ys = [p[1] for p in tradeoff_pts]
# 简单线性拟合
if len(xs) >= 2:
    coeffs = np.polyfit(xs, ys, 1)
    x_line = np.linspace(min(xs) - 0.01, max(xs) + 0.01, 50)
    y_line = np.polyval(coeffs, x_line)
    ax.plot(x_line, y_line, 'r--', alpha=0.3, linewidth=1.5, zorder=1)
    # 标注斜率
    slope = coeffs[0]
    mid_x = np.mean(xs)
    mid_y = np.polyval(coeffs, mid_x)
    ax.annotate(f"1:8 trade-off\nslope={slope:.2f}", (mid_x, mid_y),
                xytext=(mid_x + 0.02, mid_y - 0.015),
                fontsize=8, color='red', alpha=0.6,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.4))

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
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#FF4444',
           markersize=14, markeredgecolor='darkred', label='FC-SB T11 (Ours)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

# 6. 轴标签和样式
ax.set_xlabel("1 - LPIPS (内容保真度 Content Fidelity ↑)", fontsize=12, fontweight='bold')
ax.set_ylabel("CLIP-S (风格转移强度 Style Transfer ↑)", fontsize=12, fontweight='bold')
ax.set_title("FC-SB vs All Baselines: Style-Content Trade-off\n(HF CLIP ViT-B/32, LPIPS-Alex, 512px, distinct5)",
             fontsize=13, fontweight='bold')

ax.set_xlim(0.25, 1.02)
ax.set_ylim(0.55, 0.85)
ax.grid(True, alpha=0.3, linestyle='-')

# 添加 Pareto 前沿方向标注
ax.annotate("Pareto最优\n(高风格+高保真)", xy=(0.95, 0.83),
            fontsize=9, color='green', fontweight='bold', alpha=0.6,
            ha='center')

plt.tight_layout()
plt.savefig("docs/72/pareto_scatter_all_baselines.png", dpi=200, bbox_inches='tight')
plt.savefig("docs/72/pareto_scatter_all_baselines.pdf", bbox_inches='tight')
print("Saved: docs/72/pareto_scatter_all_baselines.png/pdf")
