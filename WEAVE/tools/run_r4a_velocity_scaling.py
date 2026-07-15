#!/usr/bin/env python
"""
R4-A: Velocity Magnitude Scaling Experiment

Hypothesis: FM loss + implicit regularization causes v_pred to be systematically too small,
leading to z_1 not reaching target → gray/foggy outputs.

This script tests different velocity scaling factors at inference time (zero training changes!)
- Loads R2-B checkpoint (best so far with anti-whitening)
- Runs eval with velocity_scale = [1.0, 1.5, 2.0, 3.0]
- Saves summary_grid.png and metrics for each scale
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import gc
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from PIL import Image

# Add src to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tools -> project root
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config_schema import ExperimentConfig, load_experiment_config
from model import build_model_from_config
from style_families import prune_state_dict_for_tokenizer_family
from utils.training import strip_compile_prefix
from utils.inference import LGTInference, decode_latent, load_vae
from utils.dataset import AdaCUTLatentDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_with_velocity_scale(
    checkpoint_path: str,
    device: str = "cuda",
    velocity_scale: float = 1.0,
):
    """Load model from checkpoint and wrap with velocity scaling."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    raw_config = checkpoint.get("config", {}) or {}
    config = ExperimentConfig.from_mapping(raw_config)

    # Build model
    model = build_model_from_config(
        config.model,
        bridge_cfg=config.bridge,
        use_checkpointing=False,
    ).to(device)

    # Load state dict
    state_dict = strip_compile_prefix(checkpoint["model_state_dict"])
    state_dict, removed = prune_state_dict_for_tokenizer_family(
        state_dict,
        tokenizer_family=str(getattr(config.model, "tokenizer_family", "legacy_factorized")),
        contract_family=str(getattr(config.model, "contract_family", "legacy")),
        style_injection_mode=str(getattr(config.model, "style_injection_mode", "none")),
        proximal_mode=str(getattr(config.model, "proximal_mode", "off")),
        style_delta_mode=str(getattr(config.model, "style_delta_mode", "none")),
        output_appearance_alignment_mode=str(getattr(config.model, "output_appearance_alignment_mode", "none")),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Store velocity_scale as attribute for later use
    model.velocity_scale = velocity_scale

    logger.info(f"Model loaded with velocity_scale={velocity_scale}")
    return model, config


class VelocityScaledInference:
    """Wrapper around LGTInference that passes velocity_scale to model.forward()"""

    def __init__(self, base_inference: LGTInference, velocity_scale: float):
        self.base_inference = base_inference
        self.velocity_scale = velocity_scale
        self.model = base_inference.model
        self.device = base_inference.device

        # Monkey-patch the model's forward to accept velocity_scale
        original_forward = self.model.forward

        def scaled_forward(*args, **kwargs):
            kwargs['velocity_scale'] = self.velocity_scale
            return original_forward(*args, **kwargs)

        self.model.forward = scaled_forward

    def transfer_style(self, *args, **kwargs):
        """Delegate to base inference with scaled velocity"""
        return self.base_inference.transfer_style(*args, **kwargs)


def run_eval_with_scale(
    checkpoint_path: str,
    output_dir: Path,
    velocity_scale: float,
    test_image_dir: str,
    cache_dir: str,
    num_samples: int = 20,
    num_steps: int = 8,
    batch_size: int = 2,
):
    """Run evaluation with a specific velocity scale factor."""

    scale_label = f"scale_{velocity_scale:.1f}".replace(".", "_")
    scale_output_dir = output_dir / scale_label
    scale_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"R4-A Eval: velocity_scale={velocity_scale} -> {scale_output_dir}")
    logger.info(f"{'='*60}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, config = load_model_with_velocity_scale(checkpoint_path, device, velocity_scale)

    # Create base inference (without scaling)
    base_inference = LGTInference(
        model_path=checkpoint_path,
        device=device,
        num_steps=num_steps,
    )

    # Wrap with velocity scaling
    inference = VelocityScaledInference(base_inference, velocity_scale)

    # Load VAE
    vae = load_vae(config, device)
    if vae is None:
        logger.error("Failed to load VAE")
        return None

    # Load test dataset
    data_cfg = config.data
    dataset = AdaCUTLatentDataset(
        data_root=test_image_dir,
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=False,
        batch_size_hint=batch_size,
        balance_target_styles_per_batch=True,
        latent_cache_mode="packed",
        latent_cache_dir=data_cfg.latent_cache_dir,
        device=device,
    )

    style_names = dataset.style_subdirs
    num_styles = len(style_names)
    logger.info(f"Styles: {style_names}")

    # Generate samples
    results = []
    images_saved = 0

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    logger.info(f"Generating {num_samples} samples with batch_size={batch_size}")

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if images_saved >= num_samples:
                break

            content = batch["content"].to(device)
            target_style_id = batch["target_style_id"]
            source_style_id = batch.get("source_style_id")

            # Get target style latent
            target_style = batch["target_style"].to(device)

            # Run style transfer
            try:
                output_latent = inference.transfer_style(
                    content,
                    target_style_latent=target_style,
                    target_style_id=target_style_id,
                    num_steps=num_steps,
                )

                # Decode to image
                output_image = decode_latent(output_latent, vae, config)
                content_image = decode_latent(content, vae, config)
                target_style_image = decode_latent(target_style, vae, config)

                # Save individual images
                for i in range(output_image.shape[0]):
                    if images_saved >= num_samples:
                        break

                    src_style_name = style_names[source_style_id[i].item()] if source_style_id is not None else "unknown"
                    tgt_style_name = style_names[target_style_id[i].item()]

                    # Save generated image
                    out_img = output_image[i].cpu()
                    out_path = scale_output_dir / "images" / f"{src_style_name}_to_{tgt_style_name}_{images_saved:03d}.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    save_pil_image(out_img, out_path)

                    # Save source and target for reference
                    src_img = content_image[i].cpu()
                    tgt_img = target_style_image[i].cpu()
                    save_pil_image(src_img, scale_output_dir / "images" / f"src_{images_saved:03d}.png")
                    save_pil_image(tgt_img, scale_output_dir / "images" / f"tgt_{images_saved:03d}.png")

                    results.append({
                        "source_style": src_style_name,
                        "target_style": tgt_style_name,
                        "output_path": str(out_path),
                        "velocity_scale": velocity_scale,
                    })

                    images_saved += 1

                if batch_idx % 2 == 0:
                    logger.info(f"  Generated {images_saved}/{num_samples} images...")

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    elapsed = time.time() - start_time
    logger.info(f"Generated {images_saved} images in {elapsed:.1f}s ({elapsed/max(images_saved,1):.1f}s/image)")

    # Create summary grid
    if results:
        create_summary_grid(scale_output_dir, results, velocity_scale)

    # Collect latent statistics for R4-B diagnosis
    collect_latency_stats(model, dataset, device, scale_output_dir, velocity_scale, num_batches=5)

    # Clean up
    del model, inference, base_inference, vae
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "velocity_scale": velocity_scale,
        "num_images": images_saved,
        "output_dir": str(scale_output_dir),
        "time_sec": elapsed,
        "results": results,
    }


def save_pil_image(tensor: torch.Tensor, path: Path):
    """Save a tensor [C, H, W] in [0,1] as PNG image."""
    img = tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def create_summary_grid(output_dir: Path, results: list, velocity_scale: float):
    """Create a summary grid showing source -> generated transfers."""
    images_dir = output_dir / "images"
    if not images_dir.exists():
        return

    # Collect sample images for grid
    sample_size = min(8, len(results))
    samples = results[:sample_size]

    if not samples:
        return

    # Load first image to get size
    first_img = Image.open(samples[0]["output_path"])
    img_w, img_h = first_img.size

    # Create grid: 3 rows x sample_size cols (source, target, generated)
    grid = Image.new('RGB', (img_w * sample_size, img_h * 3), color='white')

    for idx, sample in enumerate(samples):
        src_path = images_dir / f"src_{idx:03d}.png"
        tgt_path = images_dir / f"tgt_{idx:03d}.png"
        gen_path = Path(sample["output_path"])

        if src_path.exists():
            grid.paste(Image.open(src_path), (idx * img_w, 0))
        if tgt_path.exists():
            grid.paste(Image.open(tgt_path), (idx * img_w, img_h))
        if gen_path.exists():
            grid.paste(Image.open(gen_path), (idx * img_w, img_h * 2))

    # Save grid
    grid_path = output_dir / f"summary_grid_scale{velocity_scale:.1f}.png".replace(".", "_")
    grid.save(grid_path)
    logger.info(f"Saved summary grid: {grid_path}")


@torch.no_grad()
def collect_latency_stats(model, dataset, device, output_dir: Path, velocity_scale: float, num_batches: int = 5):
    """Collect latent space statistics for R4-B diagnosis.

    Compare z_1_hat (model output) vs y_proj (target) to localize fog/whitening.
    """
    logger.info(f"\nCollecting latent stats for velocity_scale={velocity_scale}...")

    stats = {
        "velocity_scale": velocity_scale,
        "z_1_hat_mean": [],
        "z_1_hat_std": [],
        "z_1_hat_min": [],
        "z_1_hat_max": [],
        "y_proj_mean": [],
        "y_proj_std": [],
        "y_proj_min": [],
        "y_proj_max": [],
        "velocity_mean": [],
        "velocity_std": [],
        "endpoint_alpha": [],
    }

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    model.eval()
    batch_count = 0

    for batch in dataloader:
        if batch_count >= num_batches:
            break

        x = batch["content"].to(device)  # source latent (z_0)
        y = batch["target_style"].to(device)  # target latent (z_1)
        style_id = batch["target_style_id"]

        t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)  # t=0 for endpoint prediction

        # Forward pass with velocity scaling
        v_pred = model(
            x,
            t=t,
            style_id=style_id,
            style_latent=y,
            target_latent=y,
            velocity_scale=velocity_scale,
        )

        # Compute predicted endpoint: z_1_hat = x + (1-t) * v = x + v (when t=0)
        z_1_hat = x + v_pred

        # Collect statistics
        stats["z_1_hat_mean"].append(z_1_hat.mean().item())
        stats["z_1_hat_std"].append(z_1_hat.std().item())
        stats["z_1_hat_min"].append(z_1_hat.min().item())
        stats["z_1_hat_max"].append(z_1_hat.max().item())

        stats["y_proj_mean"].append(y.mean().item())
        stats["y_proj_std"].append(y.std().item())
        stats["y_proj_min"].append(y.min().item())
        stats["y_proj_max"].append(y.max().item())

        stats["velocity_mean"].append(v_pred.mean().item())
        stats["velocity_std"].append(v_pred.std().item())

        # Endpoint alpha: how far did we move relative to target distance?
        displacement = (z_1_hat - x).pow(2).mean().sqrt()
        target_distance = (y - x).pow(2).mean().sqrt()
        alpha = (displacement / (target_distance + 1e-6)).item()
        stats["endpoint_alpha"].append(alpha)

        batch_count += 1

    # Aggregate stats
    aggregated = {}
    for key in stats:
        if isinstance(stats[key], list) and len(stats[key]) > 0:
            values = stats[key]
            aggregated[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        else:
            aggregated[key] = stats[key]

    # Save stats
    stats_path = output_dir / f"latent_stats_scale{velocity_scale:.1f}.json".replace(".", "_")
    with open(stats_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    logger.info(f"  Latent stats saved: {stats_path}")
    logger.info(f"  z_1_hat: mean={aggregated['z_1_hat_mean']['mean']:.4f} ± {aggregated['z_1_hat_mean']['std']:.4f}")
    logger.info(f"  y_proj:  mean={aggregated['y_proj_mean']['mean']:.4f} ± {aggregated['y_proj_mean']['std']:.4f}")
    logger.info(f"  velocity: mean={aggregated['velocity_mean']['mean']:.4f} ± {aggregated['velocity_mean']['std']:.4f}")
    logger.info(f"  endpoint_alpha: mean={np.mean(stats['endpoint_alpha']):.4f} (1.0 = reached target)")

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="R4-A: Velocity Scaling Experiment")
    parser.add_argument("--checkpoint", type=str,
                        default="exp/task4_iter/r2b_with_antiwhiten/epoch_0002.pt",
                        help="Path to R2-B checkpoint")
    parser.add_argument("--output_dir", type=str,
                        default="exp/task4_iter/r4a_velocity_scaling",
                        help="Output directory for results")
    parser.add_argument("--scales", type=float, nargs='+',
                        default=[1.0, 1.5, 2.0, 3.0],
                        help="Velocity scaling factors to test")
    parser.add_argument("--test_image_dir", type=str,
                        default=None,
                        help="Override test image directory (from config if not set)")
    parser.add_argument("--cache_dir", type=str,
                        default=None,
                        help="Override cache directory")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to generate per scale")
    parser.add_argument("--num_steps", type=int, default=8,
                        help="Number of integration steps")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for generation")

    args = parser.parse_args()

    # Resolve paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parent / checkpoint_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config from checkpoint to get test dir
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint_data.get("config", {}) or {}

    test_image_dir = args.test_image_dir or config.get("data", {}).get("test_image_dir", "")
    cache_dir = args.cache_dir or config.get("training", {}).get("full_eval_cache_dir", "")

    logger.info("="*70)
    logger.info("R4-A: VELOCITY MAGNITUDE SCALING EXPERIMENT")
    logger.info("="*70)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Scales to test: {args.scales}")
    logger.info(f"Test images: {test_image_dir}")
    logger.info(f"Num samples/scale: {args.num_samples}")
    logger.info(f"Integration steps: {args.num_steps}")
    logger.info("="*70)

    # Verify checkpoint exists
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Run experiments for each scale (serial execution)
    all_results = []

    for scale in args.scales:
        result = run_eval_with_scale(
            checkpoint_path=str(checkpoint_path),
            output_dir=output_dir,
            velocity_scale=scale,
            test_image_dir=test_image_dir,
            cache_dir=cache_dir,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
        )
        if result:
            all_results.append(result)

    # Generate final comparison report
    generate_comparison_report(output_dir, all_results)

    logger.info("\n" + "="*70)
    logger.info("R4-A EXPERIMENT COMPLETE")
    logger.info("="*70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Visually compare summary grids for different scales")
    logger.info("2. Check latent stats JSON files for magnitude analysis")
    logger.info("3. If scaling helps → implement velocity_magnitude_loss in training (R4-C)")
    logger.info("="*70)


def generate_comparison_report(output_dir: Path, results: list):
    """Generate a comparison report across all scales."""

    report_lines = [
        "# R4-A: Velocity Scaling Comparison Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Results Summary\n",
        "| Scale | Images | Time (s) | Fog Level (visual) | Notes |",
        "|-------|--------|----------|-------------------|------|",
    ]

    for r in results:
        scale = r["velocity_scale"]
        n_images = r["num_images"]
        time_sec = r["time_sec"]
        report_lines.append(f"| {scale:.1f} | {n_images} | {time_sec:.1f} | TODO: visual check | {r['output_dir']} |")

    report_lines.extend([
        "\n## Latent Statistics Comparison\n",
        "Check `latent_stats_scale_X.json` files for detailed numbers.\n",
        "\n## Key Metrics to Compare\n",
        "- **endpoint_alpha**: Should approach 1.0 (reached target)",
        "- **z_1_hat std vs y_proj std**: Gap indicates under/over-shooting",
        "- **Visual quality**: Grid images show actual rendering\n",
        "\n## Hypothesis Validation\n",
        "- If higher scale → less fog → confirms velocity magnitude hypothesis",
        "- If no change → root cause is elsewhere (VAE? loss function?)",
        "- If too high scale → artifacts/oversaturation → optimal scale exists\n",
    ])

    report_path = output_dir / "R4A_COMPARISON_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"\nComparison report saved: {report_path}")


if __name__ == "__main__":
    main()
