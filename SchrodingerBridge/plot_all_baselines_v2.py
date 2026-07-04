"""
绘制包含所有baseline的散点图
横轴: 1-LPIPS (内容保持)
纵轴: CLIP-S (风格相似度)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 所有方法数据
methods = [
    # Baselines (gray)
    ("Identity", 0.6933, 0.0000, "baseline"),
    ("AdaIN", 0.6679, 0.7425, "baseline"),
    ("WCT", 0.7063, 0.6348, "baseline"),
    ("SD-Turbo", 0.6933, 0.0033, "baseline"),
    ("SDEdit-0.35", 0.7797, 0.4508, "baseline"),
    ("SDEdit-0.40", 0.7934, 0.4826, "baseline"),
    ("StyleID", 0.8223, 0.5523, "baseline"),
    ("CUT", 0.7137, 0.3743, "baseline"),
    ("SaMST", 0.6183, 0.7490, "baseline"),
    
    # External methods (purple)
    ("SaMam", 0.7175, 0.2423, "external"),
    ("Seedream", 0.7198, 0.4767, "external"),
    
    # Ours (blue star)
    ("FC-SB (Ours)", 0.7213, 0.2868, "ours"),
    
    # Ablations (orange)
    ("Structure-aligned", 0.7037, 0.2722, "ablation"),
    ("LL mean AdaIN", 0.7215, 0.3856, "ablation"),
    ("LL std AdaIN", 0.7172, 0.2889, "ablation"),
    ("LL cov WCT", 0.7205, 0.3069, "ablation"),
    ("Depth=2", 0.7172, 0.2705, "ablation"),
    ("Dim=32", 0.7153, 0.2705, "ablation"),
    ("Gate=0", 0.7080, 0.2641, "ablation"),
    
    # Failed methods (red)
    ("Plan A α=0.5", 0.7286, 0.3747, "failed"),
    ("Plan A α=0.7", 0.7329, 0.4097, "failed"),
    ("Plan A α=1.0", 0.7333, 0.4636, "failed"),
    ("Plan C AdaLN", 0.7174, 0.2790, "failed"),
    ("Plan D Split", 0.7150, 0.2597, "failed"),
    ("Plan E YCbCr", 0.7200, 0.3085, "failed"),
]

# 分类颜色
colors = {
    "baseline": "#999999",
    "external": "#9467bd",
    "ours": "#1f77b4",
    "ablation": "#ff7f0e",
    "failed": "#d62728",
}

# 分类标记
markers = {
    "baseline": "o",
    "external": "D",
    "ours": "*",
    "ablation": "^",
    "failed": "x",
}

# 分类大小
sizes = {
    "baseline": 80,
    "external": 100,
    "ours": 300,
    "ablation": 90,
    "failed": 70,
}

fig, ax = plt.subplots(figsize=(10, 8))

# 绘制每个类别
for category in ["baseline", "external", "ablation", "failed", "ours"]:
    cat_methods = [(name, clip, lpips, cat) for name, clip, lpips, cat in methods if cat == category]
    if not cat_methods:
        continue
    
    names = [m[0] for m in cat_methods]
    clips = [m[1] for m in cat_methods]
    lpips = [m[2] for m in cat_methods]
    x = [1 - l for l in lpips]
    
    ax.scatter(x, clips, c=colors[category], marker=markers[category], 
               s=sizes[category], label=category.capitalize(), alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # 标注名称（只标注ours和关键方法）
    for i, name in enumerate(names):
        if category == "ours" or name in ["Identity", "StyleID", "SaMam", "SDEdit-0.35"]:
            ax.annotate(name, (x[i], clips[i]), textcoords="offset points", 
                       xytext=(5, 5), fontsize=8, ha='left')

# 绘制Pareto前沿
pareto_points = [(1-0.0000, 0.6933), (1-0.2423, 0.7175), (1-0.2868, 0.7213), (1-0.4508, 0.7797), (1-0.5523, 0.8223)]
pareto_points.sort()
pareto_x = [p[0] for p in pareto_points]
pareto_y = [p[1] for p in pareto_points]
ax.plot(pareto_x, pareto_y, 'g--', linewidth=2, alpha=0.5, label='Pareto Frontier')

ax.set_xlabel('1 - LPIPS (Content Preservation)', fontsize=12, fontweight='bold')
ax.set_ylabel('CLIP-S (Style Similarity)', fontsize=12, fontweight='bold')
ax.set_title('Style-Content Trade-off: All Methods Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower right', fontsize=10)

# 设置坐标轴范围
ax.set_xlim(0.4, 1.05)
ax.set_ylim(0.60, 0.85)

plt.tight_layout()
plt.savefig('aaai2027_v2/fig_all_baselines_scatter_v2.png', dpi=300, bbox_inches='tight')
plt.savefig('aaai2027_v2/fig_all_baselines_scatter_v2.pdf', bbox_inches='tight')
print("✓ 散点图已保存: fig_all_baselines_scatter_v2.png/pdf")

# 打印Pareto最优点
print("\nPareto最优点:")
print("1. Identity: clip=0.6933, lpips=0.0000 (完美内容，无风格)")
print("2. SaMam: clip=0.7175, lpips=0.2423 (高内容，中等风格)")
print("3. FC-SB (Ours): clip=0.7213, lpips=0.2868 (平衡)")
print("4. SDEdit-0.35: clip=0.7797, lpips=0.4508 (高风格，中等内容)")
print("5. StyleID: clip=0.8223, lpips=0.5523 (最高风格，差内容)")
