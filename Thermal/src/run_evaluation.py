"""
LGT Evaluation Pro: Optimized with Persistent Feature Caching & Double Sampling
Target Hardware: RTX 4070 Laptop (8GB VRAM) | CPU: 7940HX
"""

import argparse
import json
import os
from pathlib import Path
import torch

# 🔥 Enable Tensor Cores for float32 matrix multiplication (Fixes UserWarning)
torch.set_float32_matmul_precision('high')

import numpy as np
import csv
import random
import time
import hashlib
from tqdm import tqdm
from collections import defaultdict
from PIL import Image

# Metrics
try:
    import lpips
except ImportError:
    lpips = None

import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

# Project imports
from inference import LGTInference, load_vae, encode_image, decode_latent

# ==========================================
# Optimized Feature Extractors
# ==========================================

class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        # Load VGG only once, freeze immediately
        vgg = models.vgg16(pretrained=True).features.eval().to(device)
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.layer_ids = [8, 15] 
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1))

    def get_features(self, x):
        # Normalize and extract
        x = (x.to(self.mean.device) - self.mean) / self.std
        feats = []
        h = x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.layer_ids:
                feats.append(h.detach().cpu()) # Store on CPU to save VRAM
        return feats

def compute_distance_cpu_gpu_hybrid(feats_gen_gpu, feats_ref_cpu, device):
    """Compute MSE between GPU features and CPU cached features efficiently."""
    dists = []
    for f_gen, f_ref in zip(feats_gen_gpu, feats_ref_cpu):
        f_ref_gpu = f_ref.to(device)
        d = F.mse_loss(f_gen, f_ref_gpu, reduction='mean')
        dists.append(d.item())
    return np.mean(dists)

def to_lpips_input(img_tensor):
    return img_tensor * 2.0 - 1.0

# ==========================================
# Main Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default="../eval_cache", help="Directory to store shared feature caches")
    parser.add_argument('--num_steps', type=int, default=15)
    parser.add_argument('--max_src_samples', type=int, default=30, help="Max source images to generate (user specified)")
    parser.add_argument('--max_ref_compare', type=int, default=50, help="Randomly sample X refs for metric calculation (speedup)")
    parser.add_argument('--batch_size', type=int, default=24, help="Batch size for inference to fit in 8GB VRAM")
    parser.add_argument('--force_regen', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Paths & Config
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists(): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    
    # Resolve Test Data Path
    test_dir_raw = args.test_dir if args.test_dir else cfg.get('training', {}).get('test_image_dir', '')
    test_dir = Path(test_dir_raw)
    if not test_dir.exists():
        raise ValueError(f"Test directory not found: {test_dir}")

    style_subdirs = cfg.get('data', {}).get('style_subdirs', [])
    if not style_subdirs:
        # Fallback: auto-detect subdirs
        style_subdirs = [d.name for d in test_dir.iterdir() if d.is_dir()]
    
    test_images = {}
    for style_id, style_name in enumerate(style_subdirs):
        s_dir = test_dir / style_name
        if not s_dir.exists(): continue
        # Only take valid images
        images = sorted([p for p in s_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.webp']])
        test_images[style_id] = (style_name, images)

    # 2. Load Models
    print(f"⚡ Loading Models (Device: {device})...")
    lgt = LGTInference(str(checkpoint_path), device=device, num_steps=args.num_steps)
    vae = load_vae(device)
    vgg_extractor = VGGFeatureExtractor(device=device)
    loss_fn = lpips.LPIPS(net='vgg', verbose=False).to(device) if lpips else None
    to_pil = ToPILImage()

    has_clip = False
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        has_clip = True
        print("✓ CLIP Loaded")
    except:
        print("⚠️ CLIP not found, skipping text/style metrics")

    # ==========================================
    # 3. Persistent Feature Caching
    # ==========================================
    # Generate unique hash based on test_dir path to identify the dataset
    dataset_hash = hashlib.md5(str(test_dir.resolve()).encode()).hexdigest()[:8]
    cache_file = cache_dir / f"ref_feats_{dataset_hash}.pt"

    ref_features = {}
    
    if cache_file.exists() and not args.force_regen:
        print(f"📦 Found feature cache: {cache_file}")
        try:
            ref_features = torch.load(cache_file, map_location='cpu', weights_only=False)
            print("✓ Cache loaded successfully")
        except Exception as e:
            print(f"⚠️ Cache load failed ({e}), re-computing...")
    
    if not ref_features:
        print("\n🏗️  Computing Reference Features (One-time setup)...")
        for style_id, (style_name, img_list) in test_images.items():
            # Cache ALL images (or a large enough subset) to be safe
            # We sample from this pool later
            ref_features[style_id] = []
            
            pbar = tqdm(img_list, desc=f"Featurizing {style_name}")
            for img_path in pbar:
                try:
                    img_pil = Image.open(img_path).convert('RGB').resize((256,256))
                    img_t = T.ToTensor()(img_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        # VGG
                        v_feats = vgg_extractor.get_features(img_t)
                        # CLIP
                        c_emb = None
                        if has_clip:
                            inputs = clip_processor(images=to_pil(img_t.squeeze(0).cpu()), return_tensors='pt').to(device)
                            c_emb = clip_model.get_image_features(**inputs)
                            c_emb = (c_emb / c_emb.norm(p=2, dim=-1, keepdim=True)).cpu()
                            
                    ref_features[style_id].append({
                        'path': str(img_path),
                        'vgg': v_feats, 
                        'clip': c_emb
                    })
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
        
        torch.save(ref_features, cache_file)
        print(f"✓ Cache saved to {cache_file}")

    # ==========================================
    # 4. Evaluation Loop
    # ==========================================
    csv_path = out_dir / 'metrics.csv'
    # Resume check
    processed_keys = set()
    if csv_path.exists() and not args.force_regen:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_keys.add((row['src_image'], row['tgt_style']))
    
    csv_mode = 'a' if csv_path.exists() else 'w'
    csv_file = open(csv_path, csv_mode, newline='')
    columns = ['src_style', 'tgt_style', 'src_image', 'gen_image', 'content_lpips', 'style_lpips', 'clip_style', 'clip_content']
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    if csv_mode == 'w': writer.writeheader()

    print(f"\n🚀 Starting Evaluation")
    print(f"   Mode: {args.max_src_samples} sources x {args.max_ref_compare} refs/metric")

    # 1. 收集所有待处理的源图信息
    all_src_info = []
    for s_id, (s_name, s_list) in test_images.items():
        rng = random.Random(42)
        sampled = s_list[:]
        rng.shuffle(sampled)
        for p in sampled[:args.max_src_samples]:
            all_src_info.append({'path': p, 'style_id': s_id, 'style_name': s_name})

    # 2. 分批次处理 (Chunked Processing for 8GB VRAM)
    num_src_total = len(all_src_info)
    num_styles = len(style_subdirs)
    batch_size = args.batch_size

    for b_start in range(0, num_src_total, batch_size):
        b_end = min(b_start + batch_size, num_src_total)
        batch_info = all_src_info[b_start:b_end]
        print(f"\n📦 Processing Source Batch {b_start//batch_size + 1}/{(num_src_total-1)//batch_size + 1} (Size: {len(batch_info)})")

        # Load batch images
        src_tensors = []
        for item in batch_info:
            img = Image.open(item['path']).convert('RGB').resize((256, 256))
            src_tensors.append(T.ToTensor()(img))
        
        src_batch = torch.stack(src_tensors).to(device)
        src_style_ids = torch.tensor([item['style_id'] for item in batch_info], device=device)
        
        with torch.no_grad():
            # Batch Inversion
            latents_src = encode_image(vae, src_batch, device).to(torch.float32)
            latents_x0 = lgt.inversion(latents_src, src_style_ids)
            
            # Batch CLIP for sources
            src_clips = []
            if has_clip:
                for i in range(len(batch_info)):
                    pil_src = to_pil(src_batch[i].cpu())
                    inputs = clip_processor(images=pil_src, return_tensors='pt').to(device)
                    c_emb = clip_model.get_image_features(**inputs)
                    src_clips.append(c_emb / c_emb.norm(p=2, dim=-1, keepdim=True))
                src_clips = torch.cat(src_clips)

            # Generate for each target style
            for tgt_id in range(num_styles):
                tgt_name = style_subdirs[tgt_id]
                print(f"  → Generating target style: {tgt_name}")
                
                # Batch Generation
                tgt_ids = torch.full((len(batch_info),), tgt_id, device=device, dtype=torch.long)
                latents_gen = lgt.generation(latents_x0, tgt_ids)
                imgs_gen = decode_latent(vae, latents_gen, device)
                
                # Batch CLIP for generated images
                gen_clips = None
                if has_clip:
                    pil_gens = [to_pil(imgs_gen[i].cpu()) for i in range(len(batch_info))]
                    inputs = clip_processor(images=pil_gens, return_tensors='pt').to(device)
                    gen_clips = clip_model.get_image_features(**inputs)
                    gen_clips = gen_clips / gen_clips.norm(p=2, dim=-1, keepdim=True)

                # 3. 指标计算与保存
                for i in range(len(batch_info)):
                    src_item = batch_info[i]
                    out_name = f"{src_item['style_name']}_{src_item['path'].stem}_to_{tgt_name}.jpg"
                    out_path = out_dir / out_name
                    
                    # 保存图像
                    from torchvision.utils import save_image
                    save_image(imgs_gen[i], out_path)

                    # 内容指标 (LPIPS & CLIP)
                    c_lpips = 0.0
                    if loss_fn:
                        c_lpips = float(loss_fn(to_lpips_input(imgs_gen[i:i+1]), to_lpips_input(src_batch[i:i+1])).item())
                    
                    c_clip_score = 0.0
                    if has_clip:
                        c_clip_score = float(F.cosine_similarity(gen_clips[i:i+1], src_clips[i:i+1]).item())

                    # 风格指标 (从缓存中采样参考图)
                    tgt_refs = ref_features[tgt_id]
                    current_refs = random.sample(tgt_refs, min(len(tgt_refs), args.max_ref_compare))
                    
                    s_lpips_vals = []
                    s_clip_vals = []
                    
                    for ref in current_refs:
                        if loss_fn:
                            ref_img = T.ToTensor()(Image.open(ref['path']).convert('RGB').resize((256,256))).unsqueeze(0).to(device)
                            val = loss_fn(to_lpips_input(imgs_gen[i:i+1]), to_lpips_input(ref_img)).item()
                            s_lpips_vals.append(val)
                        
                        if has_clip and ref['clip'] is not None:
                            val = F.cosine_similarity(gen_clips[i:i+1], ref['clip'].to(device)).item()
                            s_clip_vals.append(val)

                    s_lpips = np.mean(s_lpips_vals) if s_lpips_vals else 0.0
                    s_clip = np.mean(s_clip_vals) if s_clip_vals else 0.0

                    # 写入 CSV
                    writer.writerow({
                        'src_style': src_item['style_name'],
                        'tgt_style': tgt_name,
                        'src_image': src_item['path'].name,
                        'gen_image': out_name,
                        'content_lpips': c_lpips,
                        'style_lpips': s_lpips,
                        'clip_style': s_clip,
                        'clip_content': c_clip_score
                    })
            
            csv_file.flush()
            torch.cuda.empty_cache() # 每个 Batch 处理完后清理显存

    csv_file.close()
    
    # ==========================================
    # 5. Generate Summary JSON
    # ==========================================
    generate_summary_json(csv_path, out_dir, checkpoint_path)

def generate_summary_json(csv_path, out_dir, ckpt_path):
    print("\n📊 Generating Summary...")
    rows = []
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader: rows.append(r)
            
    if not rows: return

    def to_f(x): return float(x) if x else 0.0

    # Matrix Aggregation
    matrix = defaultdict(lambda: defaultdict(list))
    for r in rows:
        matrix[r['src_style']][r['tgt_style']].append(r)

    matrix_json = {}
    transfer_pool = []
    identity_pool = []
    photo_transfer_pool = []

    for src, targets in matrix.items():
        matrix_json[src] = {}
        for tgt, items in targets.items():
            stats = {
                'count': len(items),
                'clip_style': np.mean([to_f(x['clip_style']) for x in items]),
                'style_lpips': np.mean([to_f(x['style_lpips']) for x in items]),
                'content_lpips': np.mean([to_f(x['content_lpips']) for x in items]),
                'clip_content': np.mean([to_f(x.get('clip_content', 0)) for x in items])
            }
            matrix_json[src][tgt] = stats
            
            if src == tgt:
                identity_pool.append(stats)
            else:
                transfer_pool.append(stats)
                if src == 'photo':
                    photo_transfer_pool.append(stats)

    def pool_avg(pool, key):
        if not pool: return 0.0
        return float(np.mean([x[key] for x in pool]))

    summary = {
        'checkpoint': str(ckpt_path),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'matrix_breakdown': matrix_json,
        'analysis': {
            'style_transfer_ability': {
                'clip_style': pool_avg(transfer_pool, 'clip_style'),
                'content_lpips': pool_avg(transfer_pool, 'content_lpips')
            },
            'photo_to_art_performance': {
                'clip_style': pool_avg(photo_transfer_pool, 'clip_style'),
                'valid': len(photo_transfer_pool) > 0
            }
        }
    }
    
    sum_path = out_dir / 'summary.json'
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved: {sum_path}")

if __name__ == '__main__':
    main()