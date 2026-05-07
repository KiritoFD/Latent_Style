from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _bootstrap_src() -> Path:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return root


ROOT = _bootstrap_src()

from schrodinger_bridge.utils.inference import (  # noqa: E402
    LGTInference,
    decode_latent,
    encode_image,
    load_vae,
)


def _load_config(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return dict(checkpoint.get("config", {}))


def _style_names(config: dict) -> list[str]:
    return list(config.get("data", {}).get("style_subdirs", [])) or ["photo", "Hayao", "monet", "vangogh", "cezanne"]


def _resolve_target_style(target: str, style_names: list[str]) -> tuple[int, str]:
    if target.isdigit():
        style_id = int(target)
        if style_id < 0 or style_id >= len(style_names):
            raise ValueError(f"target_style_id out of range: {style_id}")
        return style_id, style_names[style_id]
    if target not in style_names:
        raise ValueError(f"Unknown target style '{target}'. Choices: {', '.join(style_names)}")
    return style_names.index(target), target


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _load_image_tensor(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor * 2.0 - 1.0


def _save_tensor_image(tensor: torch.Tensor, output_path: Path) -> None:
    array = (
        tensor.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .byte()
        .numpy()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output_path, quality=95)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch inference for SchrodingerBridge.")
    parser.add_argument("checkpoint", type=str, help="Checkpoint path.")
    parser.add_argument("input_path", type=str, help="Single image or directory of images.")
    parser.add_argument("output_path", type=str, help="Output image path or directory.")
    parser.add_argument("--target_style", type=str, required=True, help="Target style name or id.")
    parser.add_argument("--style_adapter", type=str, default="", help="Optional external style adapter (.pt).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_steps", type=int, default=None, help="Override inference steps.")
    parser.add_argument("--step_size", type=float, default=None, help="Override ODE step size.")
    parser.add_argument("--style_strength", type=float, default=None, help="Override style strength.")
    parser.add_argument("--size", type=int, default=256, help="Resize input images to this resolution.")
    parser.add_argument("--vae_cache_dir", type=str, default=str((ROOT.parent / "Cycle-NCE" / "eval_cache" / "hf").resolve()))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    config = _load_config(checkpoint_path)
    style_names = _style_names(config)
    target_style_id, target_style_name = _resolve_target_style(str(args.target_style), style_names)
    default_num_steps = int(config.get("inference", {}).get("num_steps", 12))

    images = _collect_images(input_path)
    if not images:
        raise RuntimeError(f"No input images found under {input_path}")

    device = str(args.device)
    vae = load_vae(device=device, cache_dir=str(Path(args.vae_cache_dir).expanduser().resolve()))
    infer = LGTInference(
        str(checkpoint_path),
        device=device,
        num_steps=int(args.num_steps) if args.num_steps is not None else default_num_steps,
        step_size=args.step_size,
        style_strength=args.style_strength,
        style_adapter_path=(args.style_adapter or None),
    )

    model_scale = float(getattr(infer.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)

    is_single_output_file = input_path.is_file() and output_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    progress = tqdm(images, desc=f"Infer -> {target_style_name}")
    for image_path in progress:
        image_tensor = _load_image_tensor(image_path, size=int(args.size)).unsqueeze(0).to(device)
        latent = encode_image(vae, image_tensor, device=device)
        if abs(scale_in - 1.0) > 1e-4:
            latent = latent * scale_in
        z_out = infer.transfer_style(latent, target_style_id=target_style_id)
        if abs(scale_out - 1.0) > 1e-4:
            z_out = z_out * scale_out
        out = decode_latent(vae, z_out, device=device).squeeze(0)

        if is_single_output_file:
            final_path = output_path
        else:
            output_dir = output_path if output_path.suffix == "" else output_path.parent
            final_path = output_dir / f"{image_path.stem}_to_{target_style_name}.jpg"
        _save_tensor_image(out, final_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
