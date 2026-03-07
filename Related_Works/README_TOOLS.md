# CSV 散点图查看器 & Summary History 导入工具

这个工具集提供两个主要功能：
1. **CSV 散点图查看器** (`csv_scatter_viewer.html`) - 交互式可视化工具
2. **Summary History 导入脚本** (`import_summary_history_to_csv.py`) - 数据转换工具

## 🎯 快速开始

### 1. CSV 散点图查看器 (HTML)

#### 使用方法
1. 在浏览器中打开 `csv_scatter_viewer.html`
2. 粘贴 CSV 数据到文本框
3. 点击 **Parse CSV** 按钮
4. 选择 X 轴和 Y 轴的列
5. 可选：选择分组颜色、标签、倒数 Y 轴等

#### 主要功能
- ✅ 即时解析 CSV 数据
- ✅ 多列支持，自动检测数值列
- ✅ 交互式散点图（缩放、平移、悬停查看）
- ✅ 按分类列着色
- ✅ 数据标签和工具提示
- ✅ Y 轴倒数变换（1/y）
- ✅ 支持异常值过滤（最小 |X| 值）
- ✅ 实时统计信息

#### 示例 CSV 格式
```csv
experiment_id,epoch,transfer_clip_style,transfer_content_lpips,group
exp1,10,0.55,0.42,baseline
exp1,20,0.65,0.41,baseline
exp2,10,0.60,0.40,improved
exp2,20,0.68,0.38,improved
```

### 2. Summary History 导入脚本

#### 功能
- ✅ 从 `summary_history.json` 提取每个 epoch 的指标
- ✅ 支持单个文件或整个目录
- ✅ 递归搜索子目录中的 JSON 文件
- ✅ 自动去重（基于 experiment_id + epoch + source_file）
- ✅ 增量更新（追加新记录到已有 CSV）
- ✅ 自动列排序和扁平化

#### 安装依赖
```bash
# 无需额外依赖，使用 Python 标准库
python --version  # Python 3.7+
```

#### 使用方法

**单个文件：**
```bash
python import_summary_history_to_csv.py --input summary_history.json --output metrics.csv
```

**目录（非递归）：**
```bash
python import_summary_history_to_csv.py --input ./runs --output metrics.csv
```

**目录（递归）：**
```bash
python import_summary_history_to_csv.py --input ./runs --output metrics.csv --recursive
```

**增量更新：**
```bash
# 运行两次，第二次只会追加新记录，去重现有记录
python import_summary_history_to_csv.py --input ./runs --output metrics.csv --recursive
python import_summary_history_to_csv.py --input ./new_runs --output metrics.csv --recursive
```

**详细日志：**
```bash
python import_summary_history_to_csv.py --input ./runs --output metrics.csv -r -v
```

#### 输入 JSON 格式

脚本期望的 `summary_history.json` 结构：
```json
{
  "num_rounds": 1,
  "updated_at": "2026-03-07 23:23:34",
  "rounds": [
    {
      "epoch": 20,
      "transfer_clip_style": 0.6552714234093825,
      "transfer_content_lpips": 0.41690402363333334,
      "transfer_fid": 291.2200097518538,
      "transfer_art_fid": 418.3039224233603,
      "transfer_classifier_acc": 0.0,
      "photo_to_art_clip_style": 0.6324137675265471,
      "photo_to_art_fid": 291.2200097518538,
      "photo_to_art_art_fid": 418.3039224233603,
      "photo_to_art_classifier_acc": 0.0,
      "summary_path": "..."
    }
  ]
}
```

#### 输出 CSV 格式

生成的 CSV 包含以下列（自动排序）：
- `experiment_id` - 实验标识符（从目录或文件名提取）
- `epoch` - 训练轮数
- `source_file` - 来源 JSON 文件路径
- `updated_at` - 更新时间戳
- `transfer_clip_style` - CLIP 风格转移分数
- `transfer_content_lpips` - 内容 LPIPS 损失
- `transfer_fid` - 风格转移 FID
- `transfer_art_fid` - 艺术作品 FID
- `transfer_classifier_acc` - 分类器准确率
- `photo_to_art_clip_style` - 照片到艺术 CLIP 分数
- `photo_to_art_fid` - 照片到艺术 FID
- `photo_to_art_art_fid` - 照片到艺术的艺术 FID
- `photo_to_art_classifier_acc` - 照片到艺术分类准确率
- 其他自定义字段

## 📊 完整工作流示例

### 场景 1：可视化现有 CSV
```bash
# 用浏览器打开
start csv_scatter_viewer.html

# 粘贴已有的 runs_eval_summary.csv 内容
# 选择 X = "sta_clip_style", Y = "sta_content_lpips"
# 勾选 "Invert Y" 来倒数处理 LPIPS
# 点击 Parse CSV 和 Plot 即可看到散点图
```

### 场景 2：从实验结果生成 CSV 然后可视化
```bash
# 第一步：从多个实验的 summary_history.json 生成 CSV
python import_summary_history_to_csv.py --input ./runs --output all_experiments.csv -r -v

# 第二步：在浏览器中打开生成的 CSV
# 复制 all_experiments.csv 的内容到 csv_scatter_viewer.html
# 可视化看看 transfer_clip_style vs transfer_content_lpips

# 第三步：添加外部基准
# 也可以把 runs_eval_summary.csv 合并进来
```

### 场景 3：持续追踪多个实验
```bash
# 初始化
python import_summary_history_to_csv.py --input ./cut_5x5 --output tracker.csv -r

# 训练完成，添加新实验
python import_summary_history_to_csv.py --input ./sdedit_multi --output tracker.csv -r

# 训练又完成了，新增数据
python import_summary_history_to_csv.py --input ./sdturbo_5x5 --output tracker.csv -r

# 所有 epoch 的所有指标都在一个 CSV 中，可以直接粘贴到 HTML 查看器
```

## 🔧 脚本选项

### import_summary_history_to_csv.py

```
Options:
  --input, -i       必需。输入文件或目录路径
  --output, -o      必需。输出 CSV 文件路径
  --recursive, -r   递归搜索子目录中的 JSON 文件
  --verbose, -v     打印详细日志
  --help, -h        显示帮助信息
```

## 💡 使用技巧

### Tip 1：过滤异常值
在 HTML 查看器中使用"Min Absolute X Value"来排除异常点：
- 如果 clip_style 有很小的值，设置 `0.001` 来过滤它们
- 这在处理失败或不完整的实验时很有用

### Tip 2：颜色分组
按"group"列着色来可视化实验类别：
- 所有相同颜色的点表示同一组实验
- 易于比较不同方法的表现

### Tip 3：标签注释
选择"run_id"或"experiment_id"添加标签，可以：
- 快速识别特定的数据点
- 在报告中引用特定实验

### Tip 4：倒数变换
勾选"Invert Y"来看 `1/content_lpips`：
- 更高的值通常更好（而负 LPIPS 越低越好）
- 使图表更直观

## 🐛 故障排除

### 问题：CSV 解析失败
**原因：** CSV 格式不正确（引号、逗号、换行符）
**解决：** 
- 确保列值用逗号分隔
- 如果值包含逗号，用双引号包裹
- 避免在值中使用实际的换行符

### 问题：图表没有显示
**原因：** 没有有效的数值列
**解决：**
- 检查选定的 X/Y 列是否都是数值
- 查看浏览器控制台是否有错误信息

### 问题：CSV 记录重复
**原因：** 脚本去重失败
**解决：**
- 运行脚本时使用 `-v` 查看去重日志
- 检查 experiment_id 和 epoch 是否正确提取

### 问题：找不到 JSON 文件
**原因：** 文件名或路径不匹配
**解决：**
- 确保文件名以 `summary_history` 开头且后缀为 `.json`
- 使用 `-r` 开启递归搜索
- 使用 `-v` 查看找到的文件列表

## 开发者说明

### CSV 查看器的技术栈
- HTML5 + CSS3（现代响应式设计）
- Plotly.js（交互式图表库）
- 纯 JavaScript（无依赖）

### 脚本的技术特点
- 纯 Python 标准库（无外部依赖）
- 支持 CSV DictReader/DictWriter 的所有标准功能
- 自动处理字段值的扁平化
- 高效的去重算法（O(n) 时间复杂度）

## 📝 许可和归属

这些工具是为数据处理和可视化设计的通用脚本。

---

**最后更新**: 2026-03-07
**版本**: 1.0
