from __future__ import annotations

import argparse
from pathlib import Path

import torch

from runtime import Ablate43FastInference


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, pipeline: torch.nn.Module, step_size: float, style_strength: float) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.step_size = float(step_size)
        self.style_strength = float(style_strength)

    def forward(self, image_nchw_neg1_1: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        return self.pipeline(image_nchw_neg1_1, style_id, self.step_size, self.style_strength)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Ablate43 full pipeline (VAE+Core+VAE) to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=Path("../Cycle-NCE/Ablate43/Ablate43_S01_Baseline_Gold/epoch_0060.pt"))
    parser.add_argument("--model-py", type=Path, default=Path("../Cycle-NCE/Ablate43/Ablate43_S01_Baseline_Gold/model.py"))
    parser.add_argument("--output", type=Path, default=Path("./ablate43_full_pipeline.onnx"))
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--style-strength", type=float, default=1.0)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch/height/width axes")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = Ablate43FastInference(
        ckpt=args.checkpoint,
        model_py=args.model_py,
        vae_id=args.vae,
        device=device,
        dtype="fp32",
        use_compile=False,
    )

    wrapper = OnnxExportWrapper(engine.pipeline, args.step_size, args.style_strength).to(engine.device)
    wrapper.eval()

    h = max(8, (int(args.height) // 8) * 8)
    w = max(8, (int(args.width) // 8) * 8)
    b = max(1, int(args.batch))

    dummy_image = torch.randn((b, 3, h, w), device=engine.device, dtype=torch.float32)
    dummy_style = torch.zeros((b,), device=engine.device, dtype=torch.long)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "image": {0: "batch", 2: "height", 3: "width"},
            "style_id": {0: "batch"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_style),
            str(args.output),
            input_names=["image", "style_id"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=int(args.opset),
            do_constant_folding=True,
        )

    print(f"[ok] ONNX exported: {args.output.resolve()}")


if __name__ == "__main__":
    main()
