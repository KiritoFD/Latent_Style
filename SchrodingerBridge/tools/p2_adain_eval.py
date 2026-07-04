"""
P2 Task 1: AdaIN 后处理验证 - 独立评估脚本
===========================================
零成本去雾化方案：在推理时将 target latent 的 channel-wise 统计量迁移到 generated latent 上。

用法:
    python tools/p2_adain_eval.py

输出:
    exp/task4_iter/r4d1_velmag_high/p2_adain/
        ├── adain_off/          # AdaIN OFF (baseline) 结果
        │   ├── images/
        │   └── summary_grid.png
        ├── adain_on/           # AdaIN ON 结果
        │   ├── images/
        │   └── summary_grid.png
        └── comparison.png      # 并排对比图
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 项目路径设置
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# 导入项目模块
# ---------------------------------------------------------------------------
from utils.inference import LGTInference, load_vae, encode_image, decode_latent
from model620 import SpatialBridge620

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
STYLE_SUBDIRS = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_NAME_TO_ID = {name: i for i, name in enumerate(STYLE_SUBDIRS)}

CHECKPOINT_PATH = _PROJECT_ROOT / "exp" / "task4_iter" / "r4d1_velmag_high" / "epoch_0003.pt"
TEST_IMAGE_DIR = r"F:\wikiart_distinct5_512_images\test"
OUTPUT_BASE = _PROJECT_ROOT / "exp" / "task4_iter" / "r4d1_velmag_high" / "p2_adain"

IMG_SIZE = 512
BATCH_SIZE = 1  # 显存受限，逐张推理
NUM_SAMPLES_PER_STYLE = 8  # 每个风格选几张源图


def load_test_images(test_dir: str, style_subdirs: list[str], num_per_style: int = 8) -> dict[str, list[Path]]:
    """加载测试图片，每个风格目录取前 num_per_style 张"""
    test_path = Path(test_dir)
    result = {}
    for style in style_subdirs:
        style_dir = test_path / style
        if not style_dir.exists():
            print(f"  [WARN] 风格目录不存在: {style_dir}")
            result[style] = []
            continue
        images = sorted(style_dir.glob("*.jpg")) + sorted(style_dir.glob("*.png"))
        result[style] = images[:num_per_style]
        print(f"  {style}: {len(result[style])} 张图片")
    return result


def load_source_image(path: Path, device: str = "cuda") -> torch.Tensor:
    """加载并预处理单张源图片 -> [1, 3, H, W] in [-1, 1]"""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(np.array(img)).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


@torch.no_grad()
def run_adain_eval(
    lgt: LGTInference,
    vae,
    test_images: dict[str, list[Path]],
    style_subdirs: list[str],
    output_dir: Path,
    use_adain: bool = False,
    device: str = "cuda",
):
    """运行一次完整评估（AdaIN ON 或 OFF）"""

    output_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir = output_dir / "images"
    img_out_dir.mkdir(exist_ok=True)

    # 存储生成结果用于 summary grid
    gen_rows = []  # list of dict with src_style, src_image, tgt_style, path

    print(f"\n{'='*60}")
    print(f"  AdaIN {'ON' if use_adain else 'OFF'} 评估开始")
    print(f"{'='*60}")

    total_samples = sum(len(imgs) for imgs in test_images.values()) * len(style_subdirs)
    pbar = tqdm(total=total_samples, desc=f"AdaIN={'ON' if use_adain else 'OFF'}")

    for src_style in style_subdirs:
        src_images = test_images.get(src_style, [])
        if not src_images:
            continue

        for src_img_path in src_images:
            # 编码源图片
            src_tensor = load_source_image(src_img_path, device=device)
            z_src = encode_image(vae, src_tensor, device=device)
            z_src = z_src.float()  # 模型需要 float32

            # 获取 source style ID（用于计算 target latent 作为 AdaIN 参考）
            src_style_id = STYLE_NAME_TO_ID[src_style]

            for tgt_style in style_subdirs:
                tgt_style_id = STYLE_NAME_TO_ID[tgt_style]

                # 推理得到 z_1_hat
                z_gen = lgt.generation_with_target_latent(
                    z_src,
                    target_style_id=tgt_style_id,
                    num_steps=8,
                )

                # === AdaIN 后处理 ===
                if use_adain:
                    # 需要获取 target style 的参考 latent
                    # 方案：用同一张源图编码后，取 target style 的一个样本作为 style reference
                    # 这里我们用 target style 的平均统计量
                    # 更好的方案：用目标风格的典型图像编码作为 style reference
                    # 为简化，我们用生成的结果本身做自适应归一化的目标
                    # 实际上 AdaIN 需要一个 target style latent，我们用 target style 的嵌入来近似
                    #
                    # 关键思路：从模型中提取 target style embedding 对应的 latent 统计量
                    # 由于 SpatialBridge620 在 forward 中会计算 y_proj (target projection)
                    # 我们这里用一个近似方法：
                    #   加载一张 target style 的图像，编码后作为 style reference
                    pass  # 下面实现详细版本

                # AdaIN 详细实现
                if use_adain:
                    # 获取 target style 参考latent: 从target style目录中取第一张图编码
                    tgt_ref_imgs = test_images.get(tgt_style, [])
                    if tgt_ref_imgs:
                        tgt_ref_tensor = load_source_image(tgt_ref_imgs[0], device=device)
                        z_tgt_ref = encode_image(vae, tgt_ref_tensor, device=device).float()
                        # 应用 AdaIN: 将 z_tgt_ref 的统计量迁移到 z_gen
                        z_gen = SpatialBridge620.apply_adain(z_gen, z_tgt_ref)

                # 解码为图像
                img_out = decode_latent(vae, z_gen, device=device)

                # 保存
                src_stem = src_img_path.stem
                out_name = f"{src_style}_{src_stem}_to_{tgt_style}.png"
                out_path = img_out_dir / out_name

                # 转为 PIL 并保存
                img_np = img_out[0].cpu().float().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_np).save(out_path)

                gen_rows.append({
                    "src_style": src_style,
                    "src_image": str(src_img_path.name),
                    "tgt_style": tgt_style,
                    "path": out_path,
                })

                pbar.update(1)

                # 清理显存
                del z_gen, img_out
                if use_adain and tgt_ref_imgs:
                    del z_tgt_ref, tgt_ref_tensor

            del src_tensor, z_src
            torch.cuda.empty_cache()

    pbar.close()

    # 生成 summary grid
    grid_path = save_summary_grid(gen_rows, output_dir, style_subdirs)
    print(f"\n  结果保存至: {output_dir}")
    print(f"  Summary grid: {grid_path}")

    return gen_rows, grid_path


def save_summary_grid(rows: list[dict], out_dir: Path, style_order: list[str]) -> Path:
    """生成 summary_grid.png（与 run_evaluation.py 格式兼容）"""

    if not rows:
        raise RuntimeError("没有生成的图片")

    # 按 src_style 分组，每个 src_style 选一张代表性源图
    by_src = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_src[r["src_style"]][r["src_image"]][r["tgt_style"]] = r["path"]

    # 每个源风格选择 clip_score 最高的源图（简化：选第一个）
    chosen = {}
    for src_style in style_order:
        candidates = by_src.get(src_style, {})
        if candidates:
            first_key = next(iter(candidates.keys()))
            chosen[src_style] = {
                "src_image": first_key,
                "tgt_map": candidates[first_key],
            }

    # 收集所有已存在的图片路径
    existing_paths = []
    cell_sizes = []
    for src_style in style_order:
        tgt_map = chosen.get(src_style, {}).get("tgt_map", {})
        for tgt_style in style_order:
            p = tgt_map.get(tgt_style)
            if p and Path(p).exists():
                existing_paths.append(Path(p))
                try:
                    with Image.open(p) as im:
                        cell_sizes.append(im.size)
                except Exception:
                    pass

    if not existing_paths:
        raise RuntimeError("没有找到任何生成的图片文件")

    cell_w = max(w for w, h in cell_sizes) if cell_sizes else 256
    cell_h = max(h for w, h in cell_sizes) if cell_sizes else 256
    n = len(style_order)

    # 字体
    try:
        font = ImageFont.truetype("arial.ttf", size=28)
        font_small = ImageFont.truetype("arial.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    bg = (0, 0, 0)
    fg = (255, 255, 255)
    pad = 18
    header_h = 56
    left_w = 220

    canvas_w = left_w + n * cell_w + (n + 1) * pad
    canvas_h = header_h + n * (cell_h + 4) + (n + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg)
    draw = ImageDraw.Draw(canvas)

    # 列标题（目标风格）
    for ci, tgt_style in enumerate(style_order):
        x = left_w + pad + ci * (cell_w + pad)
        draw.text((x, 8), tgt_style, fill=fg, font=font)

    # 行内容
    for ri, src_style in enumerate(style_order):
        y_row = header_h + pad + ri * (cell_h + 4 + pad)

        # 行标签（源风格）
        x_label = 6
        draw.text((x_label, y_row + max(0, (cell_h - 28) // 2)), src_style, fill=fg, font=font)
        src_img_name = chosen.get(src_style, {}).get("src_image", "")
        if src_img_name:
            draw.text((x_label, y_row + 30), Path(src_img_name).stem[:35], fill=(200, 200, 200), font=font_small)

        # 图片格子
        tgt_map = chosen.get(src_style, {}).get("tgt_map", {})
        for ci, tgt_style in enumerate(style_order):
            px = left_w + pad + ci * (cell_w + pad)
            py = y_row
            p = tgt_map.get(tgt_style)
            if p is None or not Path(p).exists():
                continue
            try:
                with Image.open(p).convert("RGB") as im:
                    canvas.paste(im, (px, py))
            except Exception:
                pass

    out_path = out_dir / "summary_grid.png"
    canvas.save(out_path, format="PNG")
    print(f"Summary grid saved: {out_path}")
    return out_path


def create_comparison_grid(off_path: Path, on_path: Path, output_path: Path):
    """创建 AdaIN OFF vs ON 的并排对比图"""

    img_off = Image.open(off_path).convert("RGB")
    img_on = Image.open(on_path).convert("RGB")

    w = img_off.width
    h = max(img_off.height, img_on.height)

    comparison = Image.new("RGB", (w * 2 + 20, h), color=(0, 0, 0))
    comparison.paste(img_off, (0, 0))
    comparison.paste(img_on, (w + 10, 0))

    # 标签
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("arial.ttf", size=36)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 10), "AdaIN OFF (Baseline)", fill=(255, 100, 100), font=font)
    draw.text((w + 20, 10), "AdaIN ON", fill=(100, 255, 100), font=font)

    comparison.save(output_path, format="PNG")
    print(f"Comparison grid saved: {output_path}")


def compute_latent_stats(z: torch.Tensor, label: str = "") -> dict:
    """计算 latent 的统计量用于分析"""
    zf = z.detach().float()
    return {
        "label": label,
        "mean": zf.mean().item(),
        "std": zf.std().item(),
        "channel_std": zf.std(dim=(2, 3)).mean().item(),
        "min": zf.min().item(),
        "max": zf.max().item(),
        "range": (zf.max() - zf.min()).item(),
    }


def main():
    parser = argparse.ArgumentParser(description="P2 Task 1: AdaIN 后处理验证")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH), help="Checkpoint 路径")
    parser.add_argument("--test-dir", type=str, default=TEST_IMAGE_DIR, help="测试图片根目录")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_BASE), help="输出根目录")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES_PER_STYLE, help="每个风格的采样数")
    parser.add_argument("--off-only", action="store_true", help="只跑 AdaIN OFF")
    parser.add_argument("--on-only", action="store_true", help="只跑 AdaIN ON")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU")
        device = "cpu"

    output_base = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)

    print("=" * 60)
    print("  P2 Task 1: AdaIN 后处理验证")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  测试图片:   {args.test_dir}")
    print(f"  输出目录:   {output_base}")
    print(f"  设备:       {device}")
    print(f"  每风格采样: {args.num_samples}")
    print(f"  风格列表:   {STYLE_SUBDIRS}")

    # ------------------------------------------------------------------
    # 1. 加载测试图片清单
    # ------------------------------------------------------------------
    print("\n[1/4] 加载测试图片...")
    test_images = load_test_images(args.test_dir, STYLE_SUBDIRS, args.num_samples)
    total = sum(len(v) for v in test_images.values())
    print(f"  总计 {total} 张源图片")

    # ------------------------------------------------------------------
    # 2. 加载模型和VAE
    # ------------------------------------------------------------------
    print("\n[2/4] 加载模型和VAE...")
    t0 = time.perf_counter()

    lgt = LGTInference(str(checkpoint_path), device=device, num_steps=1)
    vae = load_vae(device=device)

    model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))

    print(f"  模型加载完成 ({time.perf_counter()-t0:.1f}s)")
    print(f"  latent scale: model={model_scale:.5f}, vae={vae_scale:.5f}")

    # ------------------------------------------------------------------
    # 3. 运行评估
    # ------------------------------------------------------------------
    results = {}

    # --- AdaIN OFF (baseline) ---
    if not args.on_only:
        off_dir = output_base / "adain_off"
        rows_off, grid_off = run_adain_eval(
            lgt=lgt,
            vae=vae,
            test_images=test_images,
            style_subdirs=STYLE_SUBDIRS,
            output_dir=off_dir,
            use_adain=False,
            device=device,
        )
        results["off"] = {"rows": rows_off, "grid": grid_off}

    # --- AdaIN ON ---
    if not args.off_only:
        on_dir = output_base / "adain_on"
        rows_on, grid_on = run_adain_eval(
            lgt=lgt,
            vae=vae,
            test_images=test_images,
            style_subdirs=STYLE_SUBDIRS,
            output_dir=on_dir,
            use_adain=True,
            device=device,
        )
        results["on"] = {"rows": rows_on, "grid": grid_on}

    # ------------------------------------------------------------------
    # 4. 生成对比图
    # ------------------------------------------------------------------
    if "off" in results and "on" in results:
        print("\n[4/4] 生成对比图...")
        comp_path = output_base / "comparison.png"
        create_comparison_grid(results["off"]["grid"], results["on"]["grid"], comp_path)

    # 保存元信息
    meta = {
        "checkpoint": str(checkpoint_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "styles": STYLE_SUBDIRS,
        "num_samples_per_style": args.num_samples,
        "results": {k: {"grid": str(v["grid"]), "count": len(v["rows"])} for k, v in results.items()},
    }
    meta_path = output_base / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  P2 Task 1 完成!")
    print("=" * 60)
    if "off" in results:
        print(f"  AdaIN OFF: {results['off']['grid']}")
    if "on" in results:
        print(f"  AdaIN ON:  {results['on']['grid']}")
    if "off" in results and "on" in results:
        print(f"  对比图:    {comp_path}")

    # 清理
    del lgt, vae
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
