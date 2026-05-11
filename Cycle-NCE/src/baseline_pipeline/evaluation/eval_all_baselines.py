import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import lpips
from cmmd import compute_cmmd
import open_clip
import json
from pathlib import Path
import hashlib
import gc

# Configuration
IMAGE_SIZE = 256
BATCH_SIZE = 4  # Optimized for 8GB VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
CACHE_DIR = "../eval_cache"
CLIP_CACHE_DIR = os.path.join(CACHE_DIR, "clip_features")
os.makedirs(CLIP_CACHE_DIR, exist_ok=True)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                        std=[0.26862954, 0.26130258, 0.27577711])
])

# Feature caching utilities
def get_cache_key(dir_path, model_name="clip_vitb32"):
    """Generate unique cache key for a directory"""
    files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    hash_str = hashlib.md5("".join(files).encode()).hexdigest()[:16]
    return f"{os.path.basename(dir_path)}_{model_name}_{hash_str}.pt"

def load_cached_features(cache_key):
    """Load cached features if available"""
    cache_path = os.path.join(CLIP_CACHE_DIR, cache_key)
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=DEVICE)
    return None

def save_cached_features(features, cache_key):
    """Save features to cache"""
    cache_path = os.path.join(CLIP_CACHE_DIR, cache_key)
    torch.save(features, cache_path)

def _extract_clip_embeddings(output):
    """Robust CLIP embedding extraction from different output types"""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, 'image_embeds') and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
    if isinstance(output, dict):
        if 'image_embeds' in output: return output['image_embeds']
        if 'pooler_output' in output: return output['pooler_output']
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError(f"Could not extract embeddings from CLIP output of type {type(output)}")

# Load models once
print("Loading evaluation models...")
# LPIPS
lpips_model = lpips.LPIPS(net='alex').to(DEVICE, dtype=DTYPE)
lpips_model.eval()

# DINO v2
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE, dtype=DTYPE)
dino_model.eval()

# CLIP - Use same local cache as main project
cache_dir = Path(CACHE_DIR).resolve()
hf_cache_dir = cache_dir / "hf"
os.environ["HF_HOME"] = str(hf_cache_dir)
os.environ["HF_HUB_CACHE"] = str(hf_cache_dir / "hub")
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Use local cache only

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="openai",
    device=DEVICE,
    cache_dir=str(hf_cache_dir)
)
clip_model.eval()

def load_images(dir_path, transform_fn):
    images = []
    paths = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for path in paths:
        img = Image.open(os.path.join(dir_path, path)).convert('RGB')
        images.append(transform_fn(img))
    return torch.stack(images).to(DEVICE, dtype=DTYPE), paths

def compute_lpips(content_imgs, generated_imgs):
    scores = []
    for i in range(0, len(content_imgs), BATCH_SIZE):
        batch_c = content_imgs[i:i+BATCH_SIZE]
        batch_g = generated_imgs[i:i+BATCH_SIZE]
        with torch.no_grad():
            score = lpips_model(batch_c, batch_g).squeeze()
        scores.extend(score.cpu().numpy().tolist())
    return sum(scores) / len(scores)

def compute_dino_struct(content_imgs, generated_imgs):
    scores = []
    for i in range(0, len(content_imgs), BATCH_SIZE):
        batch_c = content_imgs[i:i+BATCH_SIZE]
        batch_g = generated_imgs[i:i+BATCH_SIZE]
        with torch.no_grad():
            feat_c = dino_model(batch_c, is_training=False)
            feat_g = dino_model(batch_g, is_training=False)
            sim = F.cosine_similarity(feat_c, feat_g, dim=1)
        scores.extend(sim.cpu().numpy().tolist())
    return sum(scores) / len(scores)

@torch.no_grad()
def extract_clip_features(images):
    """Extract CLIP features with caching and memory optimization"""
    features = []
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i+BATCH_SIZE]
        try:
            with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                feat = clip_model.encode_image(batch)
                feat = F.normalize(feat, dim=-1)
            features.append(feat.detach())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                gc.collect()
                torch.cuda.empty_cache()
                # Process one by one if OOM
                for single_img in batch:
                    single_img = single_img.unsqueeze(0)
                    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                        feat = clip_model.encode_image(single_img)
                        feat = F.normalize(feat, dim=-1)
                    features.append(feat.detach())
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise e
    return torch.cat(features, dim=0)

def compute_clip_sim(generated_dir, reference_dir):
    """Compute CLIP similarity with caching"""
    # Check cache for reference features
    ref_cache_key = get_cache_key(reference_dir)
    ref_feats = load_cached_features(ref_cache_key)
    
    if ref_feats is None:
        # Load and extract reference features
        ref_imgs, _ = load_images(reference_dir, clip_transform)
        ref_feats = extract_clip_features(ref_imgs)
        save_cached_features(ref_feats, ref_cache_key)
    
    # Extract generated features
    gen_imgs, _ = load_images(generated_dir, clip_transform)
    gen_feats = extract_clip_features(gen_imgs)
    
    # Compute mean similarity (each generated vs all references)
    sim_matrix = gen_feats @ ref_feats.T
    mean_sim = sim_matrix.mean().item()
    
    return mean_sim

def evaluate_baseline(baseline_name, style_name):
    print(f"\nEvaluating {baseline_name} - {style_name}...")
    result_dir = f"../results/{baseline_name}/{style_name}"
    content_dir = "../../../style_data/overfit50/photo"  # Use existing overfit50 test set
    style_dir = f"../../../style_data/overfit50/{style_name}"  # Use existing style test set
    
    # Load images for LPIPS and DINO
    content_imgs, _ = load_images(content_dir, transform)
    generated_imgs, _ = load_images(result_dir, transform)
    
    # Compute metrics
    metrics = {}
    metrics["baseline"] = baseline_name
    metrics["style"] = style_name
    metrics["resolution"] = IMAGE_SIZE
    
    # CMMD
    print("Computing CMMD...")
    metrics["cmmd"] = compute_cmmd(style_dir, result_dir, batch_size=max(1, BATCH_SIZE//2))
    gc.collect()
    torch.cuda.empty_cache()
    
    # LPIPS
    print("Computing LPIPS...")
    metrics["lpips"] = compute_lpips(content_imgs, generated_imgs)
    gc.collect()
    torch.cuda.empty_cache()
    
    # DINO Structure
    print("Computing DINO Structure...")
    metrics["dino_struct"] = compute_dino_struct(content_imgs, generated_imgs)
    del content_imgs, generated_imgs
    gc.collect()
    torch.cuda.empty_cache()
    
    # CLIP Style
    print("Computing CLIP Style Similarity...")
    metrics["clip_style"] = compute_clip_sim(result_dir, style_dir)
    gc.collect()
    torch.cuda.empty_cache()
    
    # CLIP Content
    print("Computing CLIP Content Similarity...")
    metrics["clip_content"] = compute_clip_sim(result_dir, content_dir)
    gc.collect()
    torch.cuda.empty_cache()
    
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="Baseline name to evaluate")
    parser.add_argument("--style", type=str, required=True, help="Style name to evaluate")
    parser.add_argument("--output", type=str, default="../results/metrics.csv", help="Output CSV path")
    args = parser.parse_args()
    
    metrics = evaluate_baseline(args.baseline, args.style)
    
    # Save to CSV
    if os.path.exists(args.output):
        df = pd.read_csv(args.output)
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    else:
        df = pd.DataFrame([metrics])
    
    df.to_csv(args.output, index=False)
    print(f"\nEvaluation completed! Metrics saved to {args.output}")
    print(pd.DataFrame([metrics]).to_string(index=False))
