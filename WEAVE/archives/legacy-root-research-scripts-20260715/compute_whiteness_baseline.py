#!/usr/bin/env python3
"""
白化诊断基线指标计算脚本
计算 generated 图片 vs 预期的白化相关指标
"""

import os
import csv
import json
import numpy as np
from PIL import Image
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_image(image_path):
    """加载图片并转换为 numpy array"""
    img = Image.open(image_path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0

def compute_global_std(img):
    """计算像素值的全局标准差（衡量整体对比度）"""
    return np.std(img)

def compute_channel_std_per_image(img):
    """计算每个通道的 std（衡量通道间方差分布）"""
    channel_stds = []
    for c in range(3):  # RGB
        channel_std = np.std(img[:, :, c])
        channel_stds.append(channel_std)
    return channel_stds

def compute_channel_std_entropy(channel_stds):
    """计算各通道 std 的熵（衡量通道多样性）"""
    # 归一化为概率分布
    probs = np.array(channel_stds) / (np.sum(channel_stds) + 1e-8)
    # 计算熵
    entropy = -np.sum(probs * np.log2(probs + 1e-8))
    return entropy

def compute_hf_energy_ratio(img):
    """计算高频能量占比 = var(high_pass) / var(total)"""
    # 使用简单的拉普拉斯算子作为高通滤波器
    from scipy.ndimage import laplace

    total_var = np.var(img)

    if total_var < 1e-8:
        return 0.0

    # 对每个通道应用拉普拉斯算子
    hf_energy = 0
    for c in range(3):
        channel = img[:, :, c]
        high_pass = laplace(channel)
        hf_energy += np.var(high_pass)

    hf_energy /= 3  # 平均

    return hf_energy / (total_var + 1e-8)

def compute_brightness_mean(img):
    """计算平均亮度"""
    # 转换为灰度图
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return np.mean(gray)

def compute_saturation_mean(img):
    """计算平均饱和度（转 HSV 后的 S 通道均值）"""
    img_uint8 = (img * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    hsv = img_pil.convert('HSV')
    hsv_array = np.array(hsv)
    # S 通道是第1个通道
    s_channel = hsv_array[:, :, 1].astype(np.float32) / 255.0
    return np.mean(s_channel)

def compute_whiteness_score(metrics_dict, weights=None):
    """
    综合白化分数（加权组合）
    指标说明：
    - global_std: 越低越白化（对比度塌缩）
    - channel_std_entropy: 越低越白化（通道多样性丢失）
    - hf_energy_ratio: 越低越白化（高频细节丢失）
    - saturation_mean: 越低越白化（颜色饱和度不足）
    """
    if weights is None:
        weights = {
            'global_std': 0.25,
            'channel_std_entropy': 0.20,
            'hf_energy_ratio': 0.30,
            'saturation_mean': 0.25
        }

    # 归一化到 [0, 1] 范围（假设正常范围）
    normalized = {
        'global_std': max(0, min(1, metrics_dict['global_std'] / 0.3)),  # 正常 ~0.15-0.25
        'channel_std_entropy': max(0, min(1, metrics_dict['channel_std_entropy'] / 1.5)),  # 正常 ~1.0-1.5
        'hf_energy_ratio': max(0, min(1, metrics_dict['hf_energy_ratio'] / 0.5)),  # 正常 ~0.2-0.4
        'saturation_mean': max(0, min(1, metrics_dict['saturation_mean'] / 0.6))  # 正常 ~0.3-0.5
    }

    # 白化分数：越高表示越白化（指标值越低）
    whiteness = sum(weights[key] * (1 - normalized[key]) for key in weights.keys())

    return whiteness

def analyze_image_for_whiteness(image_path):
    """分析单张图片的白化程度"""
    try:
        img = load_image(image_path)

        metrics = {
            'global_std': compute_global_std(img),
            'channel_std_per_image': compute_channel_std_per_image(img),
            'channel_std_entropy': compute_channel_std_entropy(compute_channel_std_per_image(img)),
            'hf_energy_ratio': compute_hf_energy_ratio(img),
            'brightness_mean': compute_brightness_mean(img),
            'saturation_mean': compute_saturation_mean(img),
        }

        metrics['whiteness_score'] = compute_whiteness_score(metrics)

        return metrics

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    base_dir = r'g:\GitHub\Latent_Style\SchrodingerBridge\exp\620_spatial_bridge\620_spatial_bridge_ablation_recommended\full_eval_wfi\epoch_0001'
    images_dir = os.path.join(base_dir, 'images')
    output_csv = os.path.join(base_dir, 'whiteness_baseline_metrics.csv')

    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return

    # 收集所有生成的图片
    results = []
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]

    print(f"Found {len(image_files)} images to analyze")

    for i, img_file in enumerate(image_files):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(image_files)}...")

        img_path = os.path.join(images_dir, img_file)

        # 解析文件名获取 source 和 target
        # 格式: {source}_{artist}_title_to_{target}.png
        parts = img_file.replace('.png', '').split('_to_')
        if len(parts) != 2:
            continue

        src_part = parts[0]
        tgt_style = parts[1]

        # 提取 source style（第一个下划线之前的部分）
        src_style = src_part.split('_')[0]

        metrics = analyze_image_for_whiteness(img_path)
        if metrics is None:
            continue

        result = {
            'image_file': img_file,
            'src_style': src_style,
            'tgt_style': tgt_style,
            **metrics,
            # 将 channel_std_per_image 转换为字符串
            'channel_std_r': metrics['channel_std_per_image'][0],
            'channel_std_g': metrics['channel_std_per_image'][1],
            'channel_std_b': metrics['channel_std_per_image'][2],
        }
        del result['channel_std_per_image']

        results.append(result)

    # 保存到 CSV
    if results:
        keys = results[0].keys()
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to: {output_csv}")

        # 打印汇总统计
        print("\n=== 白化基线指标汇总 ===")
        metric_names = ['global_std', 'channel_std_entropy', 'hf_energy_ratio',
                       'brightness_mean', 'saturation_mean', 'whiteness_score']

        for metric in metric_names:
            values = [r[metric] for r in results]
            print(f"\n{metric}:")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std:  {np.std(values):.4f}")
            print(f"  Min:  {np.min(values):.4f}")
            print(f"  Max:  {np.max(values):.4f}")

        # 按 source→target 分组统计
        print("\n=== 按 Source→Target 分组的白化分数 ===")
        group_stats = {}
        for r in results:
            key = f"{r['src_style']}→{r['tgt_style']}"
            if key not in group_stats:
                group_stats[key] = []
            group_stats[key].append(r['whiteness_score'])

        # 排序并显示最严重的白化组合
        sorted_groups = sorted(group_stats.items(), key=lambda x: np.mean(x[1]), reverse=True)

        print("\n最严重的白化组合（Top 10）：")
        for i, (group, scores) in enumerate(sorted_groups[:10]):
            print(f"{i+1}. {group}: whiteness={np.mean(scores):.4f}±{np.std(scores):.4f} (n={len(scores)})")

        print("\n最轻微的白化组合（Top 5）：")
        for i, (group, scores) in enumerate(sorted_groups[-5:]):
            print(f"{i+1}. {group}: whiteness={np.mean(scores):.4f}±{np.std(scores):.4f} (n={len(scores)})")

        # 保存分组统计
        summary_data = []
        for group, scores in sorted_groups:
            summary_data.append({
                'source_target': group,
                'whiteness_mean': np.mean(scores),
                'whiteness_std': np.std(scores),
                'count': len(scores)
            })

        summary_csv = os.path.join(base_dir, 'whiteness_group_summary.csv')
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['source_target', 'whiteness_mean', 'whiteness_std', 'count'])
            writer.writeheader()
            writer.writerows(summary_data)

        print(f"\nGroup summary saved to: {summary_csv}")

if __name__ == '__main__':
    main()
