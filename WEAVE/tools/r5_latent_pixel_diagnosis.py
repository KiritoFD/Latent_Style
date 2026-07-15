#!/usr/bin/env python
"""
R5: Latent Space vs Pixel Space 诊断脚本
===========================================

核心问题：雾化发生在哪个阶段？
- A) 模型输出的 latent 本身就是灰/白的 → 问题在模型
- B) latent 看起来正常但 VAE decode 后变白 → 问题在 decode

诊断方法：
1. 提取模型输出的原始 latent (z_1_hat)
2. 计算 z_1_hat 的统计量：mean, std, min, max per channel, global_std, saturation
3. 提取 target latent (y_proj) 的同样统计量
4. 对比差异：z_1_hat 的 std 是否显著低于 y_proj？
5. 用 VAE decode 两者，对比解码后的图片

输出：
- "Latent Fog Score": z_1_hat.std() / y_proj.std()
- 保存 diagnostic 对比图
"""

from __future__ import annotations

import json
import sys
import math
import gc
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # 主项目 src
sys.path.insert(0, str(PROJECT_ROOT / "exp" / "task4_iter" / "r4d1_velmag_high" / "src"))  # R4-D1 src

from config_schema import ExperimentConfig, load_experiment_config
from model import build_model_from_config


def load_checkpoint_and_config(
    ckpt_path: Path,
    config_path: Optional[Path] = None,
) -> tuple:
    """加载 checkpoint 和配置。"""
    print(f"[R5] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 尝试从 checkpoint 中加载配置
    if config_path is None and "config" in state:
        config_data = state["config"]
        if isinstance(config_data, dict):
            config = ExperimentConfig.from_dict(config_data)
        else:
            config = load_experiment_config(config_path)
    elif config_path is not None:
        config = load_experiment_config(config_path)
    else:
        raise ValueError("No config found in checkpoint or provided")

    return state, config


def build_model_from_checkpoint(
    state: dict,
    config: ExperimentConfig,
    device: torch.device,
):
    """从 checkpoint 构建模型并加载权重。"""
    print("[R5] Building model...")
    model = build_model_from_config(
        config.model,
        bridge_cfg=config.bridge,
        use_checkpointing=False,
    ).to(device)

    # 加载模型权重
    model_state = state.get("model_state_dict", {})
    if model_state:
        # 处理 compile 前缀
        def strip_compile_prefix(state_dict: dict) -> dict:
            return {
                k.replace("_orig.", "").replace("_module.", ""): v
                for k, v in state_dict.items()
            }
        model_state = strip_compile_prefix(model_state)
        model.load_state_dict(model_state, strict=False)
        print(f"[R5] Loaded {len(model_state)} parameters")

    model.eval()
    return model


def load_vae_decoder(device: torch.device):
    """加载 VAE decoder 用于将 latent 转换为图像。"""
    try:
        from diffusers import AutoencoderKL
        vae_path = PROJECT_ROOT / "stabilityai" / "stable-diffusion-2-1-base"
        if vae_path.exists():
            vae = AutoencoderKL.from_pretrained(str(vae_path), subfolder="vae").to(device)
            print("[R5] Loaded local VAE")
            return vae
    except Exception as e:
        print(f"[R5] Failed to load local VAE: {e}")

    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            subfolder="vae",
        ).to(device)
        print("[R5] Downloaded VAE from HuggingFace")
        return vae
    except Exception as e:
        print(f"[R5] Failed to download VAE: {e}")
        return None


@torch.no_grad()
def run_diagnosis(
    model,
    vae,
    config: ExperimentConfig,
    device: torch.device,
    num_samples: int = 8,
    output_dir: Path = None,
) -> Dict:
    """
    运行 Latent vs Pixel 诊断。

    Returns:
        包含所有诊断指标的字典
    """
    from utils.dataset import AdaCUTLatentDataset

    print(f"\n{'='*60}")
    print("[R5] Starting Latent vs Pixel Diagnosis")
    print(f"{'='*60}")

    # 准备输出目录
    if output_dir is None:
        output_dir = PROJECT_ROOT / "exp" / "task4_iter" / "r5_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集（只用于获取样本）
    data_cfg = config.data
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.data_root,
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=False,
        identity_ratio=None,
        batch_size_hint=num_samples,
        balance_target_styles_per_batch=True,
        preload_to_gpu=False,
        device=str(device),
    )

    # 获取一批数据
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=num_samples,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    content = batch["content"].to(device)
    target_style = batch["target_style"].to(device)
    target_style_id = batch["target_style_id"].to(device)

    print(f"[R5] Content shape: {content.shape}")
    print(f"[R5] Target style shape: {target_style.shape}")
    print(f"[R5] Style IDs: {target_style_id.tolist()}")

    # ========== Part 1: 提取模型输出 ==========
    print("\n[R5] Running model inference...")

    # 使用 t=0 (或接近0) 来获取最终预测
    t_batch = torch.zeros(content.shape[0], device=device, dtype=content.dtype)

    # 模型预测 velocity
    pred_velocity = model(
        content,
        t=t_batch,
        style_id=target_style_id,
        style_latent=target_style,
    )

    # 计算预测的 endpoint (z_1_hat)
    # z_1 = z_0 + (1 - t) * v = content + 1.0 * pred_velocity (when t=0)
    z_1_hat = content + pred_velocity

    # Target projected latent (y_proj)
    # 从 loss 函数的逻辑中提取 projected_target
    y_proj = target_style  # 简化：使用原始 target_style

    # ========== Part 2: 计算统计量 ==========
    print("\n[R5] Computing statistics...")

    def compute_latent_stats(latent: torch.Tensor, name: str) -> Dict:
        """计算 latent 的详细统计量。"""
        latent_f = latent.float()

        stats = {
            "name": name,
            "global_mean": float(latent_f.mean().item()),
            "global_std": float(latent_f.std().item()),
            "global_min": float(latent_f.min().item()),
            "global_max": float(latent_f.max().item()),
            "abs_mean": float(latent_f.abs().mean().item()),
            "rms": float(torch.sqrt((latent_f ** 2).mean()).item()),
        }

        # Per-channel 统计
        per_channel_mean = latent_f.mean(dim=[0, 2, 3])
        per_channel_std = latent_f.std(dim=[0, 2, 3])
        per_channel_min = latent_f.amin(dim=[0, 2, 3])
        per_channel_max = latent_f.amax(dim=[0, 2, 3])

        stats["channel_means"] = per_channel_mean.tolist()
        stats["channel_stds"] = per_channel_std.tolist()
        stats["channel_mins"] = per_channel_min.tolist()
        stats["channel_maxs"] = per_channel_max.tolist()

        # Per-sample 统计
        per_sample_std = latent_f.std(dim=[1, 2, 3])
        stats["sample_stds"] = per_sample_std.tolist()
        stats["mean_sample_std"] = float(per_sample_std.mean().item())

        # Dynamic range
        dynamic_range = latent_f.amax(dim=[1, 2, 3]) - latent_f.amin(dim=[1, 2, 3])
        stats["dynamic_ranges"] = dynamic_range.tolist()
        stats["mean_dynamic_range"] = float(dynamic_range.mean().item())

        # 频域统计 (高频能量)
        lowpass = F.avg_pool2d(latent_f, kernel_size=5, stride=1, padding=2)
        high_freq = latent_f - lowpass
        hf_energy = float((high_freq ** 2).mean().item())
        lf_energy = float((lowpass ** 2).mean().item())
        stats["hf_energy"] = hf_energy
        stats["lf_energy"] = lf_energy
        stats["hf_lf_ratio"] = hf_energy / (lf_energy + 1e-8)

        return stats

    source_stats = compute_latent_stats(content, "source (z_0)")
    gen_stats = compute_latent_stats(z_1_hat, "generated (z_1_hat)")
    target_stats = compute_latent_stats(target_style, "target (y)")

    # ========== Part 3: 关键指标对比 ==========
    print("\n[R5] Computing key metrics...")

    # Latent Fog Score
    latent_fog_score = gen_stats["global_std"] / (target_stats["global_std"] + 1e-8)

    # Sample std ratio
    gen_sample_stds = torch.tensor(gen_stats["sample_stds"])
    tgt_sample_stds = torch.tensor(target_stats["sample_stds"])
    sample_std_ratio = (gen_sample_stds / (tgt_sample_stds.clamp_min(1e-8))).mean().item()

    # HF energy ratio
    hf_ratio = gen_stats["hf_energy"] / (target_stats["hf_energy"] + 1e-8)

    # Dynamic range ratio
    dr_ratio = gen_stats["mean_dynamic_range"] / (target_stats["mean_dynamic_range"] + 1e-8)

    # ========== Part 4: VAE Decode 和像素级分析 ==========
    print("\n[R5] Running VAE decode...")

    pixel_metrics = {}
    decoded_images = {}

    if vae is not None:
        def decode_to_image(latent: torch.Tensor, name: str) -> tuple:
            """Decode latent to image 并计算像素级指标。"""
            # Normalize 到 VAE 的输入范围
            latent_scaled = latent / 0.18215  # SD2.1 的 latent scale factor

            # Decode
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                decoded = vae.decode(latent_scaled).sample

            # 转换到 [0, 1]
            decoded = (decoded / 2 + 0.5).clamp(0, 1)

            # 像素级统计
            pixel_f = decoded.float()
            pixel_stats = {
                "name": name,
                "mean": float(pixel_f.mean().item()),
                "std": float(pixel_f.std().item()),
                "min": float(pixel_f.min().item()),
                "max": float(pixel_f.max().item()),
                # 饱和度估计 (HSV S channel)
                "saturation_mean": 0.0,
                "brightness_mean": float(pixel_f.mean(dim=[1, 2, 3]).mean().item()),
                "contrast": float(pixel_f.std(dim=[1, 2, 3]).mean().item()),
            }

            # 计算饱和度 (简化版)
            r, g, b = decoded[:, 0], decoded[:, 1], decoded[:, 2]
            max_val = torch.stack([r, g, b], dim=0).amax(dim=0)
            min_val = torch.stack([r, g, b], dim=0).amin(dim=0)
            saturation = (max_val - min_val) / (max_val.clamp(min=1e-6))
            pixel_stats["saturation_mean"] = float(saturation.mean().item())

            return decoded, pixel_stats

        decoded_source, source_pixel = decode_to_image(content, "decoded_source")
        decoded_gen, gen_pixel = decode_to_image(z_1_hat, "decoded_gen")
        decoded_target, target_pixel = decode_to_image(target_style, "decoded_target")

        decoded_images = {
            "source": decoded_source,
            "gen": decoded_gen,
            "target": decoded_target,
        }
        pixel_metrics = {
            "source": source_pixel,
            "gen": gen_pixel,
            "target": target_pixel,
        }

        # 像素级对比
        pixel_fog_score = gen_pixel["std"] / (target_pixel["std"] + 1e-8)
        pixel_saturation_ratio = gen_pixel["saturation_mean"] / (target_pixel["saturation_mean"] + 1e-8)
        pixel_contrast_ratio = gen_pixel["contrast"] / (target_pixel["contrast"] + 1e-8)
    else:
        pixel_fog_score = None
        pixel_saturation_ratio = None
        pixel_contrast_ratio = None

    # ========== Part 5: 汇总结果 ==========
    diagnosis_result = {
        "latent_fog_score": latent_fog_score,
        "sample_std_ratio": sample_std_ratio,
        "hf_energy_ratio": hf_ratio,
        "dynamic_range_ratio": dr_ratio,
        "pixel_fog_score": pixel_fog_score,
        "pixel_saturation_ratio": pixel_saturation_ratio,
        "pixel_contrast_ratio": pixel_contrast_ratio,
        "source_stats": source_stats,
        "gen_stats": gen_stats,
        "target_stats": target_stats,
        "pixel_metrics": pixel_metrics,
    }

    # 打印诊断报告
    print("\n" + "=" * 60)
    print("[R5] DIAGNOSIS REPORT")
    print("=" * 60)

    print("\n--- Latent Space Statistics ---")
    print(f"Source (z_0):     mean={source_stats['global_mean']:.4f}, std={source_stats['global_std']:.4f}, "
          f"range=[{source_stats['global_min']:.4f}, {source_stats['global_max']:.4f}]")
    print(f"Generated (z_1): mean={gen_stats['global_mean']:.4f}, std={gen_stats['global_std']:.4f}, "
          f"range=[{gen_stats['global_min']:.4f}, {gen_stats['global_max']:.4f}]")
    print(f"Target (y):       mean={target_stats['global_mean']:.4f}, std={target_stats['global_std']:.4f}, "
          f"range=[{target_stats['global_min']:.4f}, {target_stats['global_max']:.4f}]")

    print("\n--- Key Metrics ---")
    print(f"Latent Fog Score (gen_std/tgt_std):     {latent_fog_score:.4f}")
    print(f"Sample Std Ratio:                      {sample_std_ratio:.4f}")
    print(f"HF Energy Ratio (gen/tgt):             {hf_ratio:.4f}")
    print(f"Dynamic Range Ratio (gen/tgt):         {dr_ratio:.4f}")

    if pixel_fog_score is not None:
        print("\n--- Pixel Space Statistics ---")
        print(f"Pixel Fog Score (gen_std/tgt_std):      {pixel_fog_score:.4f}")
        print(f"Pixel Saturation Ratio (gen/tgt):      {pixel_saturation_ratio:.4f}")
        print(f"Pixel Contrast Ratio (gen/tgt):        {pixel_contrast_ratio:.4f}")

        print("\n--- Decoded Images ---")
        for name, metrics in pixel_metrics.items():
            print(f"{name:15s}: brightness={metrics['brightness_mean']:.4f}, "
                  f"contrast={metrics['contrast']:.4f}, sat={metrics['saturation_mean']:.4f}")

    print("\n--- Diagnosis Conclusion ---")
    if latent_fog_score < 0.5:
        conclusion = "SEVERE LATENT FOG: Generated latent has very low variance"
        recommendation = "→ Problem is in MODEL OUTPUT (latent space)"
    elif latent_fog_score < 0.7:
        conclusion = "MODERATE LATENT FOG: Generated latent has reduced variance"
        recommendation = "→ Problem is primarily in MODEL OUTPUT"
    elif latent_fog_score < 0.9:
        conclusion = "MILD LATENT FOG: Some variance reduction detected"
        recommendation = "→ Possible combination of latent + decode issues"
    else:
        conclusion = "NO LATENT FOG: Variance looks normal"
        if pixel_fog_score is not None and pixel_fog_score < 0.7:
            recommendation = "→ Problem is likely in VAE DECODE or color space"
        else:
            recommendation = "→ No obvious fog issue detected"

    print(f"Conclusion: {conclusion}")
    print(f"Recommendation: {recommendation}")

    # ========== Part 6: 保存可视化 ==========
    print("\n[R5] Saving visualization...")
    save_diagnostic_grid(
        content, z_1_hat, target_style,
        decoded_images,
        source_stats, gen_stats, target_stats,
        diagnosis_result,
        output_dir / "diagnostic_grid.png",
    )

    # 保存 JSON 结果
    result_path = output_dir / "diagnosis_result.json"
    # Convert tensors to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(result_path, "w") as f:
        json.dump(convert_for_json(diagnosis_result), f, indent=2)
    print(f"[R5] Saved results to {result_path}")

    return diagnosis_result


def save_diagnostic_grid(
    content: torch.Tensor,
    z_1_hat: torch.Tensor,
    target_style: torch.Tensor,
    decoded_images: dict,
    source_stats: dict,
    gen_stats: dict,
    target_stats: dict,
    diagnosis_result: dict,
    output_path: Path,
):
    """保存诊断对比图网格。"""

    num_samples = min(content.shape[0], 4)  # 只显示前4个样本

    # 创建大画布
    img_size = 256
    margin = 10
    text_height = 30

    # 行: latent 可视化 (3行: source, gen, target) + decoded images (3行) + stats
    rows = 6
    cols = num_samples + 1  # 样本 + 统计信息列

    canvas_width = cols * (img_size + margin) + margin
    canvas_height = rows * (img_size + text_height + margin) + margin + 200  # 底部额外空间放文字

    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    draw = ImageDraw.Draw(canvas)

    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        font_large = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_large = font

    def tensor_to_pil(tensor: torch.Tensor, idx: int) -> Image.Image:
        """Convert tensor slice to PIL Image."""
        t = tensor[idx].detach().cpu()

        # 如果是 4-channel latent, 取前3个channel 或转灰度
        if t.shape[0] == 4:
            # 使用 RGB 三个通道
            t = t[:3]

        # Normalize 到 [0, 1]
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)

        # 转 numpy
        if t.shape[0] == 3:
            img_array = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img_array = (t.squeeze().numpy() * 255).astype(np.uint8)
            img_array = np.stack([img_array] * 3, axis=-1)

        return Image.fromarray(img_array)

    # 绘制 latent 空间可视化
    row_labels = [
        "Source Latent (z₀)",
        "Generated Latent (ẑ₁)",
        "Target Latent (y)",
        "Decoded Source",
        "Decoded Generated",
        "Decoded Target",
    ]

    tensors_to_show = [
        content,
        z_1_hat,
        target_style,
        decoded_images.get("source"),
        decoded_images.get("gen"),
        decoded_images.get("target"),
    ]

    for row_idx, (label, tensor_list) in enumerate(zip(row_labels, tensors_to_show)):
        y_pos = margin + row_idx * (img_size + text_height + margin)

        # 绘制行标签
        draw.text((margin, y_pos), label, fill="black", font=font_large)

        for col_idx in range(num_samples):
            x_pos = margin + (col_idx + 1) * (img_size + margin)

            if tensor_list is not None:
                img = tensor_to_pil(tensor_list, col_idx)
                img = img.resize((img_size, img_size), Image.LANCZOS)
                canvas.paste(img, (x_pos, y_pos + text_height))

    # 在最后一列绘制统计信息
    stats_x = margin + (num_samples + 1) * (img_size + margin)
    stats_y = margin

    # 统计文本
    stats_text = f"""DIAGNOSIS SUMMARY
==================

Latent Statistics:
  Source:  μ={source_stats['global_mean']:.3f} σ={source_stats['global_std']:.3f}
  Gen:     μ={gen_stats['global_mean']:.3f} σ={gen_stats['global_std']:.3f}
  Target:  μ={target_stats['global_mean']:.3f} σ={target_stats['global_std']:.3f}

Key Metrics:
  Latent Fog Score:   {diagnosis_result['latent_fog_score']:.4f}
  Sample Std Ratio:   {diagnosis_result['sample_std_ratio']:.4f}
  HF Energy Ratio:    {diagnosis_result['hf_energy_ratio']:.4f}
  Dynamic Range:      {diagnosis_result['dynamic_range_ratio']:.4f}
"""

    if diagnosis_result.get('pixel_fog_score') is not None:
        stats_text += f"""
Pixel Statistics:
  Pixel Fog Score:    {diagnosis_result['pixel_fog_score']:.4f}
  Saturation Ratio:   {diagnosis_result['pixel_saturation_ratio']:.4f}
  Contrast Ratio:     {diagnosis_result['pixel_contrast_ratio']:.4f}
"""

    # 判定阈值
    lfs = diagnosis_result['latent_fog_score']
    if lfs < 0.5:
        verdict = "⚠️ SEVERE FOG IN LATENT SPACE"
    elif lfs < 0.7:
        verdict = "⚡ MODERATE FOG IN LATENT SPACE"
    elif lfs < 0.9:
        verdict = "🔍 MILD FOG - CHECK DECODE"
    else:
        verdict = "✅ NO OBVIOUS LATENT FOG"

    stats_text += f"\nVerdict: {verdict}"

    draw.text((stats_x, stats_y), stats_text, fill="black", font=font)

    # 保存
    canvas.save(output_path)
    print(f"[R5] Saved diagnostic grid to {output_path}")


def main():
    """主函数。"""
    import argparse

    parser = argparse.ArgumentParser(description="R5: Latent vs Pixel Diagnosis")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./exp/task4_iter/r4d1_velmag_high/epoch_0003.pt",
        help="Path to R4-D1 checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./exp/task4_iter/r4d1_velmag_high/config.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples to diagnose",
    )
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[R5] Device: {device}")

    # 加载 checkpoint 和配置
    ckpt_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else None

    state, config = load_checkpoint_and_config(ckpt_path, config_path)

    # 构建模型
    model = build_model_from_checkpoint(state, config, device)

    # 加载 VAE
    vae = load_vae_decoder(device)

    # 运行诊断
    output_dir = Path(args.output_dir) if args.output_dir else None
    result = run_diagnosis(
        model=model,
        vae=vae,
        config=config,
        device=device,
        num_samples=args.num_samples,
        output_dir=output_dir,
    )

    print("\n[R5] Diagnosis complete!")
    return result


if __name__ == "__main__":
    main()
