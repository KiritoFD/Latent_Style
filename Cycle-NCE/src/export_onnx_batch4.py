from __future__ import annotations

import argparse
from pathlib import Path

import torch

from infer_manual_parallel import DEFAULT_CKPT, DEFAULT_VAE, ManualParallelInference


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "onnx" / "cycle_nce_full_pipeline_b4_256.onnx"
FIXED_BATCH = 4
FIXED_HW = 256


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, engine: ManualParallelInference) -> None:
        super().__init__()
        self.model = engine.model
        self.vae = engine.vae
        self.vae_scale = float(engine.vae_scale)
        self.scale_in = float(engine.scale_in)
        self.scale_out = float(engine.scale_out)
        self.default_step_size = float(engine.default_step_size)
        self.default_style_strength = float(engine.default_style_strength)

    def forward(self, image_nchw_neg1_1: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(image_nchw_neg1_1).latent_dist.mean * self.vae_scale
        if abs(self.scale_in - 1.0) > 1e-4:
            z = z * self.scale_in
        z_out = self.model.integrate(
            z,
            style_id=style_id,
            num_steps=1,
            step_size=self.default_step_size,
            style_strength=self.default_style_strength,
        )
        if abs(self.scale_out - 1.0) > 1e-4:
            z_out = z_out * self.scale_out
        y = self.vae.decode(z_out / self.vae_scale).sample
        return torch.clamp((y + 1.0) * 0.5, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed batch=4, 256x256 full pipeline ONNX")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--vae", type=str, default=DEFAULT_VAE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = ManualParallelInference(
        checkpoint=args.checkpoint,
        vae_id=args.vae,
        device=device,
        dtype="fp32",
        tf32=False,
    )
    engine.model = engine.model.float().to(engine.device)
    engine.vae = engine.vae.float().to(engine.device)
    wrapper = OnnxExportWrapper(engine).to(engine.device)
    wrapper.eval()

    dummy_image = torch.randn((FIXED_BATCH, 3, FIXED_HW, FIXED_HW), device=engine.device, dtype=torch.float32)
    dummy_style = torch.zeros((FIXED_BATCH,), device=engine.device, dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_style),
            str(args.output),
            input_names=["image", "style_id"],
            output_names=["output"],
            opset_version=int(args.opset),
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"[ok] ONNX exported: {args.output.resolve()}")


if __name__ == "__main__":
    main()
