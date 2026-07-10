"""Evaluate WEAVE plugin experiment: CLIP-S, LPIPS, DINO-S, DINO-C.

Computes the 4 canonical metrics for a directory of generated images:
  - CLIP-S:    cosine similarity between generated and target-style reference CLIP embeddings
              (max over references, averaged)
  - LPIPS:     perceptual distance between generated and source content image
              (AlexNet backbone, matching main results table protocol)
  - DINO-S:    max cosine between DINOv2 CLS(gen) and target-style reference CLS
  - DINO-C:    cosine between DINOv2 CLS(gen) and CLS(source)

Writes:
  - metrics.csv (with columns: src_style, tgt_style, src_image, gen_image, clip_style, content_lpips)
    — compatible with src/utils/compute_dino_metrics.py for extended DINO analysis
  - summary.json with all-pairs and off-diagonal aggregates
  - per_pair.csv with per-(src,tgt) breakdown

Usage:
  python tools/eval_plugin_metrics.py \
    --image-dir exp/plugin_sd15/sd15_weave/images \
    --output exp/plugin_sd15/sd15_weave/eval \
    --test-dir "G:\GitHub\Latent_Style\Dataset\eval\distinct5_512\test"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STYLE_NAMES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 512

# CLIP normalization (OpenAI ViT-B/32)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# LPIPS normalization ([-1, 1])
LPIPS_MEAN = [0.5, 0.5, 0.5]
LPIPS_STD = [0.5, 0.5, 0.5]

# DINO normalization (ImageNet)
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_clip_model(device: str = "cuda"):
    """Load CLIP ViT-B/32 (matching main results table protocol)."""
    # Try open_clip first (same as main eval)
    try:
        import open_clip
        cache_dir = (
            Path(__file__).resolve().parents[1]
            / "eval_cache"
            / "manual_clip"
            / "openai-clip-vit-base-patch32"
        )
        if cache_dir.exists():
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=str(cache_dir)
            )
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
        model = model.to(device).eval()
        print(f"  CLIP: open_clip ViT-B-32 on {device}")
        return model, preprocess, device, "open_clip"
    except ImportError:
        pass

    # Fallback to HF transformers
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print(f"  CLIP: HF transformers ViT-B-32 on {device}")
    return model, processor, device, "hf"


def load_lpips_model(device: str = "cuda"):
    """Load LPIPS with AlexNet backbone (matching main results table)."""
    import lpips
    model = lpips.LPIPS(net="alex").to(device).eval()
    print(f"  LPIPS: AlexNet on {device}")
    return model


def load_dino_model(device: str = "cuda"):
    """Load DINOv2-small (matching canonical DINO protocol)."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    print(f"  DINO: DINOv2-small on {device}")
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.inference_mode()
def extract_clip_features(images: list[Image.Image], clip_pack, batch_size: int = 32) -> torch.Tensor:
    """Extract CLIP image features (normalized)."""
    model, preprocess, device, backend = clip_pack
    all_features = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        if backend == "open_clip":
            tensors = torch.stack([preprocess(img) for img in batch]).to(device)
            features = model.encode_image(tensors)
            features = F.normalize(features, dim=-1)
        else:
            inputs = preprocess(images=batch, return_tensors="pt", padding=True).to(device)
            outputs = model.get_image_features(**inputs)
            if hasattr(outputs, "image_embeds"):
                features = outputs.image_embeds
            elif isinstance(outputs, torch.Tensor):
                features = outputs
            else:
                features = outputs[0]
            features = F.normalize(features, dim=-1)
        all_features.append(features)

    return torch.cat(all_features, dim=0) if all_features else torch.tensor([])


@torch.inference_mode()
def extract_dino_features(
    paths: list[Path], model, device: str, batch_size: int = 8
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Extract DINOv2 CLS + penultimate patch features."""
    transform = T.Compose([
        T.Resize(224, interpolation=Image.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=DINO_MEAN, std=DINO_STD),
    ])

    cls_features = []
    patch_features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        pixels = torch.stack([
            transform(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(device)
        output = model(pixels, output_hidden_states=True)
        cls = F.normalize(output.last_hidden_state[:, 0, :].float(), dim=-1).cpu()
        patches = F.normalize(output.hidden_states[-2][:, 1:, :].float(), dim=-1).cpu()
        cls_features.append(cls)
        patch_features.extend(patches[i] for i in range(patches.shape[0]))

    return torch.cat(cls_features, dim=0), patch_features


# ---------------------------------------------------------------------------
# Image loading + parsing
# ---------------------------------------------------------------------------
def parse_filename(filename: str) -> dict | None:
    """Parse '{SrcStyle}__{src_name}__to__{TgtStyle}.png'."""
    name = Path(filename).stem
    if "__to__" not in name:
        return None
    parts = name.split("__to__")
    tgt_style = parts[-1]
    src_part = "__to__".join(parts[:-1])
    src_style = None
    for s in STYLE_NAMES:
        if src_part.startswith(s + "__"):
            src_style = s
            src_name = src_part[len(s) + 2 :]
            break
    if src_style is None:
        return None
    return {"src_style": src_style, "src_name": src_name, "tgt_style": tgt_style}


def load_generated_images(image_dir: Path) -> list[dict]:
    """Load all generated images with parsed metadata."""
    images = []
    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        meta = parse_filename(img_path.name)
        if meta is None:
            continue
        meta["path"] = str(img_path)
        meta["gen_image"] = img_path.name
        images.append(meta)
    return images


def find_source_image(src_style: str, src_name: str, test_dir: Path) -> Path | None:
    """Find the source image path in the test set."""
    style_dir = test_dir / src_style
    if not style_dir.exists():
        return None
    # Try exact filename match (with various extensions)
    for ext in IMAGE_EXTS:
        candidate = style_dir / f"{src_style}__{src_name}{ext}"
        if candidate.exists():
            return candidate
    # Try just src_name
    for ext in IMAGE_EXTS:
        candidate = style_dir / f"{src_name}{ext}"
        if candidate.exists():
            return candidate
    # Fallback: search
    for p in style_dir.iterdir():
        if src_name in p.name:
            return p
    return None


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(
    image_dir: Path,
    test_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 8,
    max_refs_per_style: int = 30,
):
    """Full evaluation: CLIP-S, LPIPS, DINO-S, DINO-C."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load generated images metadata
    gen_images = load_generated_images(image_dir)
    print(f"\nFound {len(gen_images)} generated images in {image_dir}")
    if not gen_images:
        print("ERROR: No images found!")
        return

    # 2. Load models
    print("\nLoading models...")
    clip_pack = load_clip_model(device)
    lpips_model = load_lpips_model(device)
    dino_model = load_dino_model(device)

    # 3. Load style reference images (for CLIP-S and DINO-S)
    print("\nLoading style reference images...")
    style_ref_paths = {}
    style_ref_clip = {}
    style_ref_dino_cls = {}
    for style_name in STYLE_NAMES:
        style_dir = test_dir / style_name
        if not style_dir.exists():
            print(f"  [WARN] No test dir for {style_name}")
            continue
        refs = sorted(
            p for p in style_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )[:max_refs_per_style]
        if not refs:
            continue
        style_ref_paths[style_name] = refs
        print(f"  {style_name}: {len(refs)} references")

    # Pre-compute CLIP features for style references
    print("\nComputing CLIP features for style references...")
    for style_name, refs in style_ref_paths.items():
        ref_imgs = [Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS) for p in refs]
        style_ref_clip[style_name] = extract_clip_features(ref_imgs, clip_pack, batch_size=32)
        print(f"  {style_name}: CLIP features {style_ref_clip[style_name].shape}")

    # Pre-compute DINO features for style references
    print("\nComputing DINO features for style references...")
    for style_name, refs in style_ref_paths.items():
        style_ref_dino_cls[style_name], _ = extract_dino_features(refs, dino_model, device, batch_size=8)
        print(f"  {style_name}: DINO CLS {style_ref_dino_cls[style_name].shape}")

    # 4. Evaluate each generated image
    print(f"\nEvaluating {len(gen_images)} images...")
    lpips_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(LPIPS_MEAN, LPIPS_STD),
    ])

    results = []
    # Batch the generated images for CLIP + DINO feature extraction
    gen_paths = [Path(g["path"]) for g in gen_images]

    # Extract CLIP features for all generated images (batched)
    print("  Extracting CLIP features for generated images...")
    gen_clip_features = []
    for start in tqdm(range(0, len(gen_images), batch_size), desc="CLIP"):
        batch_imgs = [
            Image.open(gen_paths[i]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            for i in range(start, min(start + batch_size, len(gen_images)))
        ]
        gen_clip_features.append(extract_clip_features(batch_imgs, clip_pack, batch_size=len(batch_imgs)))
    gen_clip_features = torch.cat(gen_clip_features, dim=0) if gen_clip_features else torch.tensor([])

    # Extract DINO features for all generated images (batched)
    print("  Extracting DINO features for generated images...")
    gen_dino_cls, _ = extract_dino_features(gen_paths, dino_model, device, batch_size=batch_size)

    # Extract DINO features for unique source images
    print("  Extracting DINO features for source images...")
    source_path_map = {}
    source_paths_unique = []
    for g in gen_images:
        sp = find_source_image(g["src_style"], g["src_name"], test_dir)
        if sp is None:
            print(f"    [WARN] Source not found: {g['src_style']}/{g['src_name']}")
            continue
        if str(sp) not in source_path_map:
            source_path_map[str(sp)] = len(source_paths_unique)
            source_paths_unique.append(sp)
    source_dino_cls, _ = extract_dino_features(source_paths_unique, dino_model, device, batch_size=batch_size)

    # Compute LPIPS + assemble metrics
    print("  Computing LPIPS + assembling metrics...")
    for idx, g in enumerate(tqdm(gen_images, desc="LPIPS")):
        gen_img = Image.open(g["path"]).convert("RGB")
        src_path = find_source_image(g["src_style"], g["src_name"], test_dir)

        # CLIP-S: max cosine with style references
        tgt_style = g["tgt_style"]
        if tgt_style in style_ref_clip and idx < gen_clip_features.shape[0]:
            sims = (gen_clip_features[idx] @ style_ref_clip[tgt_style].T).cpu()
            clip_style = sims.max().item()
        else:
            clip_style = 0.0

        # DINO-S: max cosine with style references
        if tgt_style in style_ref_dino_cls and idx < gen_dino_cls.shape[0]:
            sims = (gen_dino_cls[idx] @ style_ref_dino_cls[tgt_style].T).cpu()
            dino_style = sims.max().item()
        else:
            dino_style = 0.0

        # DINO-C: cosine with source
        sp_key = str(src_path) if src_path else None
        if sp_key and sp_key in source_path_map and idx < gen_dino_cls.shape[0]:
            src_idx = source_path_map[sp_key]
            dino_content = F.cosine_similarity(
                gen_dino_cls[idx:idx+1], source_dino_cls[src_idx:src_idx+1]
            ).item()
        else:
            dino_content = 0.0

        # LPIPS: perceptual distance to source
        if src_path:
            src_img = Image.open(src_path).convert("RGB")
            t1 = lpips_transform(gen_img).unsqueeze(0).to(device)
            t2 = lpips_transform(src_img).unsqueeze(0).to(device)
            with torch.no_grad():
                lpips_val = lpips_model(t1, t2).item()
        else:
            lpips_val = 0.0

        results.append({
            "src_style": g["src_style"],
            "tgt_style": g["tgt_style"],
            "src_image": f"{g['src_style']}__{g['src_name']}",
            "gen_image": g["gen_image"],
            "clip_style": clip_style,
            "content_lpips": lpips_val,
            "dino_style": dino_style,
            "dino_content": dino_content,
            "one_minus_lpips": 1.0 - lpips_val,
        })

    # 5. Aggregate
    all_pairs = _aggregate(results)
    off_diag = _aggregate([r for r in results if r["src_style"] != r["tgt_style"]])
    identity = _aggregate([r for r in results if r["src_style"] == r["tgt_style"]])

    summary = {
        "all_pairs": all_pairs,
        "off_diagonal": off_diag,
        "identity": identity,
        "n_images": len(results),
        "n_off_diagonal": sum(1 for r in results if r["src_style"] != r["tgt_style"]),
    }

    # 6. Save
    # metrics.csv (compatible with compute_dino_metrics.py)
    metrics_csv = output_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "src_style", "tgt_style", "src_image", "gen_image",
            "clip_style", "content_lpips", "dino_style", "dino_content",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    # per_pair.csv
    per_pair = _per_pair_table(results)
    per_pair_csv = output_dir / "per_pair.csv"
    with open(per_pair_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "src_style", "tgt_style", "n",
            "clip_style", "content_lpips", "dino_style", "dino_content", "one_minus_lpips",
        ])
        writer.writeheader()
        for row in per_pair:
            writer.writerow(row)

    # summary.json
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 7. Print
    _print_summary(summary, image_dir.name)

    return summary


def _aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    n = len(results)
    return {
        "clip_style": sum(r["clip_style"] for r in results) / n,
        "content_lpips": sum(r["content_lpips"] for r in results) / n,
        "dino_style": sum(r["dino_style"] for r in results) / n,
        "dino_content": sum(r["dino_content"] for r in results) / n,
        "one_minus_lpips": sum(r["one_minus_lpips"] for r in results) / n,
        "n": n,
    }


def _per_pair_table(results: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in results:
        key = (r["src_style"], r["tgt_style"])
        groups[key].append(r)
    table = []
    for (src, tgt), group in sorted(groups.items()):
        agg = _aggregate(group)
        table.append({
            "src_style": src,
            "tgt_style": tgt,
            "n": agg["n"],
            "clip_style": round(agg["clip_style"], 4),
            "content_lpips": round(agg["content_lpips"], 4),
            "dino_style": round(agg["dino_style"], 4),
            "dino_content": round(agg["dino_content"], 4),
            "one_minus_lpips": round(agg["one_minus_lpips"], 4),
        })
    return table


def _print_summary(summary: dict, name: str):
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY: {name}")
    print(f"{'='*70}")
    for section in ["all_pairs", "off_diagonal"]:
        agg = summary.get(section, {})
        if not agg:
            continue
        n = agg.get("n", 0)
        print(f"\n  [{section}] (n={n})")
        print(f"    CLIP-S:    {agg['clip_style']:.4f}")
        print(f"    LPIPS:     {agg['content_lpips']:.4f}   (1-LPIPS = {agg['one_minus_lpips']:.4f})")
        print(f"    DINO-S:    {agg['dino_style']:.4f}")
        print(f"    DINO-C:    {agg['dino_content']:.4f}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate WEAVE plugin experiment: CLIP-S, LPIPS, DINO-S, DINO-C"
    )
    parser.add_argument(
        "--image-dir", required=True,
        help="Directory containing generated images",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for metrics",
    )
    parser.add_argument(
        "--test-dir", default=r"G:\GitHub\Latent_Style\Dataset\eval\distinct5_512\test",
        help="D5 test set root (with style subdirs)",
    )
    parser.add_argument(
        "--device", default="cuda",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
    )
    parser.add_argument(
        "--max-refs-per-style", type=int, default=30,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output)

    print(f"Evaluation Setup:")
    print(f"  Image dir:  {image_dir}")
    print(f"  Test dir:   {test_dir}")
    print(f"  Output:     {output_dir}")

    evaluate(
        image_dir=image_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_refs_per_style=args.max_refs_per_style,
    )


if __name__ == "__main__":
    main()
