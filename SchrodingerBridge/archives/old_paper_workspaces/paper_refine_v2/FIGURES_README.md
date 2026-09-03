# Figure Generation Scripts - Organization Guide

## 文件结构

```
paper_refine_v2/
├── scripts_gen_figures.py          # 主图表生成脚本（包含所有图表函数）
├── figures_config.py                # 配置文件（数据、颜色、标题等）
│
├── figures/                         # 输出目录
│   ├── fig_framework_overview.png
│   ├── fig_quality_tradeoff.png
│   ├── fig_artifact_diagnostics.png
│   ├── fig_ablation_pareto.png
│   ├── fig_weight_sweep_summary.png
│   ├── fig_train_efficiency_pareto.png
│   ├── fig_pareto_editable.svg      # 可编辑的SVG版本
│   └── captions.json                # 所有图表的文字说明
│
├── figures/gen_scatter_*.py         # 通用的Pareto图表生成脚本
└── final/                           # 最终的图表（用于论文）
    ├── fig_framework.tikz.tex       # TikZ格式
    └── fig_quality_tradeoff.png
```

## 脚本功能

### 主脚本: `scripts_gen_figures.py`

#### 已实现的图表函数:
1. **framework_overview()** - 框架概览图（手绘式布局）
2. **quality_tradeoff()** - 质量权衡 Pareto 图 (LPIPS vs CLIP-style)
3. **artifact_diagnostics()** - 伪影诊断对比
4. **ablation_pareto()** - 消融研究 Pareto 图
5. **weight_sweep_summary()** - 权重扫描总结
6. **train_efficiency_pareto()** - 训练效率 Pareto 图 ✅ *已修复标注重叠*

### 通用工具脚本

- **gen_scatter_pareto.py** - 生成 Pareto 图 (matplotlib)
- **gen_scatter_svg.py** - 生成可编辑的 SVG 版本

## 最近的改进

### 修复1: 训练效率图标注重叠
**文件**: `scripts_gen_figures.py` (train_efficiency_pareto 函数)

**改动**:
- 从统一的偏移量改为按方法个性化的偏移量
- "Ours": 左移较大距离，避免与左侧轴线重叠
- "SaMST", "AdaIN", "S2WAT": 右移并根据位置上下调整

**配置**:
```python
TRAIN_EFFICIENCY_OFFSETS = {
    "Ours": (-500, 0.008),      # 左和上
    "SaMST": (250, 0.008),      # 右和上
    "S2WAT": (250, -0.010),     # 右和下
    "AdaIN": (250, 0.008),      # 右和上
}
```

### 改进2: 质量权衡图 - SVG 版本
**文件**: `gen_scatter_svg.py`

**特点**:
- 生成原生 SVG（而非 matplotlib 导出）
- 每个元素（圆圈、文字、轴线）都是独立可编辑的
- 可在 Inkscape、Adobe Illustrator 或在线编辑器中手动调整标注位置

**使用**:
```bash
cd figures
python gen_scatter_svg.py
# 输出: fig_pareto_editable.svg
```

## 快速使用

### 生成所有图表
```bash
python scripts_gen_figures.py
```
输出: `figures/` 目录中的所有 PNG + `captions.json`

### 生成可编辑的 Pareto 图
```bash
cd figures
python gen_scatter_svg.py
```
输出: `fig_pareto_editable.svg` (可在任何矢量编辑器中打开)

### 复制最终版本到论文目录
```bash
copy figures/fig_*.png aaai_submission/
```

## 配置修改指南

### 修改图表数据
编辑 `figures_config.py`:
```python
QUALITY_TRADEOFF_DATA = {
    'Ours e8': {'lpips': 0.451, 'clip_s': 0.7158, ...},
    ...
}
```

### 修改颜色
```python
QUALITY_COLORS = {
    'Ours e8': '#E74C3C',  # 改这里
    ...
}
```

### 修改图表标题和说明
```python
FIGURE_CAPTIONS = {
    "fig_quality_tradeoff": "新的说明文本...",
}
```

## 常见问题

### Q: 文字与圆圈重叠怎么办？
**A**: 
1. 对于简单的偏移调整: 编辑对应函数中的 `offsets` 字典
2. 对于复杂调整: 使用 SVG 版本在编辑器中手动拖动文字

### Q: 如何添加新的图表？
**A**:
1. 在 `figures_config.py` 中添加数据
2. 在 `scripts_gen_figures.py` 中创建新函数
3. 在 `main()` 中调用该函数
4. 为新图表添加标题到 `FIGURE_CAPTIONS`

### Q: 图表风格不一致？
**A**: 所有配置都在 `PLOT_CONFIG` 中统一管理，修改即可全局应用

## 下一步建议

1. 🎯 将 SVG 编辑脚本与 matplotlib 相比的性能测试
2. 📊 为所有 Pareto 图创建统一的模板函数
3. 📝 添加更多的文档字符串
4. 🔧 创建 CLI 工具来生成特定的图表
