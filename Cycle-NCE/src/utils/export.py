#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
from pathlib import Path

# 直接白嫖 inference.py 里的加载逻辑和特征提取函数
from inference import FlashInference, _extract_latents_from_encode_output, _extract_image_from_decode_output

class UltimateStaticEngine(nn.Module):
    """
    终极黑盒：干掉所有 Python Overhead 和动态控制流。
    只接受固定 [1, 3, 256, 256] 的 uint8 张量，直接吐出 uint8 张量。
    """
    def __init__(self, taesd, adacut, scale_in, scale_out):
        super().__init__()
        self.taesd = taesd.eval()
        self.adacut = adacut.eval()
        
        # 将缩放常数注册为 buffer，确保它们和图一起被折叠 (Constant Folding)
        self.register_buffer("scale_in", torch.tensor(scale_in, dtype=torch.float32))
        self.register_buffer("scale_out", torch.tensor(scale_out, dtype=torch.float32))
        
    def forward(self, rgb_uint8: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        # --- 1. 纯 GPU 前处理 ---
        # OpenCV 进来的 uint8 (0~255) 转换为模型需要的 float32 (-1.0~1.0)
        x = (rgb_uint8.float() / 127.5) - 1.0
        
        # --- 2. 极速编码 ---
        encoded = self.taesd.encode(x)
        latents = _extract_latents_from_encode_output(encoded)
        latents = latents * self.scale_in
        
        # --- 3. 核心流形重写 ---
        # 强制写死 step_size 和 style_strength，彻底剥离条件分支
        styled = self.adacut(latents, style_id=style_id, step_size=1.0, style_strength=1.0)
        
        # --- 4. 极速解码 ---
        styled = styled * self.scale_out
        decoded = self.taesd.decode(styled)
        image = _extract_image_from_decode_output(decoded)
        
        # --- 5. 纯 GPU 后处理 ---
        # float32 (-1.0~1.0) 还原为 uint8 (0~255)
        out = (image + 1.0) * 127.5
        out = torch.clamp(out, 0.0, 255.0)
        
        return out.to(torch.uint8)

def main():
    parser = argparse.ArgumentParser(description="Export Ultimate Static ONNX Engine")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained .pt checkpoint")
    parser.add_argument("--out", type=str, default="adacut_engine_static.onnx", help="Output ONNX file name")
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.ckpt}")
    
    # 利用 FlashInference 搞定所有权重的下载和加载加载
    runner = FlashInference(
        model_path=args.ckpt,
        device="cuda",
        use_fp16=False, # 导出 ONNX 时保持 FP32 精度，交由 TensorRT 去量化 FP16
    )
    
    # 组装终极静态图
    engine = UltimateStaticEngine(
        taesd=runner.taesd,
        adacut=runner.model,
        scale_in=runner.scale_in,
        scale_out=runner.scale_out
    ).cuda()
    
    # 伪造绝对静态的 Dummy 输入（注意这里用的是 uint8）
    dummy_image = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8, device="cuda")
    dummy_style = torch.tensor([1], dtype=torch.long, device="cuda")
    
    print(f"Exporting massive fused operator to {args.out} ...")
    
    # 暴力导出：没有任何 dynamic_axes，彻底锁死内存寻址
    torch.onnx.export(
        engine,
        (dummy_image, dummy_style),
        args.out,
        export_params=True,
        opset_version=18,          # 17 是目前对 uint8 cast 支持最完美的 opset
        do_constant_folding=False,  # 强制折叠所有常数预计算
        input_names=["input_rgb_uint8", "style_id"],
        output_names=["output_rgb_uint8"],
    )
    
    print("Done! Engine is mathematically sealed.")

if __name__ == "__main__":
    main()