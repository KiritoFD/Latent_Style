"""
LGT Evaluation Pro: Optimized with Pipeline Offloading, Async I/O & Vectorization
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
import gc
import time
import hashlib
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Metrics
try:
    import lpips
except ImportError:
    lpips = None

import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
# Project imports
from inference import LGTInference, load_vae, encode_image, decode_latent

# ==========================================
# Optimized Feature Extractors
# ==========================================

class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        # Load VGG only once, freeze immediately
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval().to(device)
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

def to_lpips_input(img_tensor):
    return img_tensor * 2.0 - 1.0

def save_image_task(tensor_cpu, path):
    """Async save task to avoid blocking GPU loop"""
    try:
        save_image(tensor_cpu, path)
    except Exception as e:
        print(f"Error saving {path}: {e}")

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
    parser.add_argument('--batch_size', type=int, default=24, help="Batch size increased due to offloading")
    parser.add_argument('--force_regen', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Paths & Config
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Thread Pool for Async I/O
    io_pool = ThreadPoolExecutor(max_workers=4)
    
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

    # Prepare Source List
    all_src_info = []
    for s_id, (s_name, s_list) in test_images.items():
        rng = random.Random(42)
        sampled = s_list[:]
        rng.shuffle(sampled)
        for p in sampled[:args.max_src_samples]:
            all_src_info.append({'path': p, 'style_id': s_id, 'style_name': s_name})

    # Buffer to pass data from Phase 1 to Phase 2
    # List of dicts: {'src_path': Path, 'tgt_style_name': str, 'tgt_style_id': int, 'gen_img': Tensor(CPU)}
    generated_buffer = []

    # ==========================================
    # PHASE 1: GENERATION (LGT + VAE)
    # ==========================================
    print(f"\n🚀 Phase 1: Generation (Batch Size {args.batch_size})")
    
    lgt = LGTInference(str(checkpoint_path), device=device, num_steps=args.num_steps)
    vae = load_vae(device)
    
    num_src_total = len(all_src_info)
    num_styles = len(style_subdirs)
    
    # Process in batches
    for b_start in range(0, num_src_total, args.batch_size):
        b_end = min(b_start + args.batch_size, num_src_total)
        batch_info = all_src_info[b_start:b_end]
        print(f"  Generating Batch {b_start//args.batch_size + 1}/{(num_src_total-1)//args.batch_size + 1}")

        # Load Source Images
        src_tensors = []
        for item in batch_info:
            img = Image.open(item['path']).convert('RGB').resize((256, 256))
            src_tensors.append(T.ToTensor()(img))
        
        src_batch = torch.stack(src_tensors).to(device)
        src_style_ids = torch.tensor([item['style_id'] for item in batch_info], device=device)
        
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                # Inversion
                latents_src = encode_image(vae, src_batch, device)
                latents_x0 = lgt.inversion(latents_src, src_style_ids)
                
                # Generation for each target style
                for tgt_id in range(num_styles):
                    tgt_name = style_subdirs[tgt_id]
                    tgt_ids = torch.full((len(batch_info),), tgt_id, device=device, dtype=torch.long)
                    
                    latents_gen = lgt.generation(latents_x0, tgt_ids)
                    imgs_gen = decode_latent(vae, latents_gen, device) # [B, 3, H, W]
                    
                    # Offload to CPU & Save Async
                    imgs_gen_cpu = imgs_gen.cpu()
                    
                    for i in range(len(batch_info)):
                        src_item = batch_info[i]
                        out_name = f"{src_item['style_name']}_{src_item['path'].stem}_to_{tgt_name}.jpg"
                        out_path = out_dir / out_name
                        
                        # Async Save
                        io_pool.submit(save_image_task, imgs_gen_cpu[i], out_path)
                        
                        # Store for Phase 2
                        generated_buffer.append({
                            'src_path': src_item['path'],
                            'src_style': src_item['style_name'],
                            'tgt_style_name': tgt_name,
                            'tgt_style_id': tgt_id,
                            'gen_img': imgs_gen_cpu[i], # Keep in RAM
                            'gen_name': out_name
                        })

    # Unload Generation Models
    del lgt, vae
    torch.cuda.empty_cache()
    gc.collect()
    print("  ✓ Generation Models Unloaded")

    # ==========================================
    # PHASE 2: EVALUATION (VGG + CLIP)
    # ==========================================
    print(f"\n🚀 Phase 2: Evaluation")
    
    # Load Evaluators
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
        print("  ✓ CLIP Loaded")
    except:
        print("  ⚠️ CLIP not found")

    # Prepare Reference Features (Cache)
    dataset_hash = hashlib.md5(str(test_dir.resolve()).encode()).hexdigest()[:8]
    cache_file = cache_dir / f"ref_feats_{dataset_hash}.pt"

    ref_features = {}
    
    if cache_file.exists() and not args.force_regen:
        print(f"📦 Found feature cache: {cache_file}")
        try:
            ref_features = torch.load(cache_file, map_location='cpu')
            print("  ✓ Cache loaded successfully")
        except Exception as e:
            print(f"  ⚠️ Cache load failed ({e}), re-computing...")
    
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
                    
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
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

    # Optimize Reference CLIP Features for Vectorization
    ref_clip_matrices = {} # style_id -> Tensor[N_ref, 512] (GPU)
    if has_clip:
        for sid, feats in ref_features.items():
            # Stack all clip embeddings for this style
            clips = [f['clip'] for f in feats if f['clip'] is not None]
            if clips:
                # [N, 1, 512] -> [N, 512]
                ref_clip_matrices[sid] = torch.cat(clips, dim=0).to(device, dtype=torch.bfloat16)

    csv_path = out_dir / 'metrics.csv'
    csv_mode = 'w' if args.force_regen or not csv_path.exists() else 'a'
    csv_file = open(csv_path, csv_mode, newline='')
    columns = ['src_style', 'tgt_style', 'src_image', 'gen_image', 'content_lpips', 'style_lpips', 'clip_style', 'clip_content']
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    if csv_mode == 'w': writer.writeheader()

    # Process Generated Buffer in Batches
    total_gen = len(generated_buffer)
    print(f"  Processing {total_gen} generated images...")
    
    for b_start in range(0, total_gen, args.batch_size):
        b_end = min(b_start + args.batch_size, total_gen)
        batch_items = generated_buffer[b_start:b_end]
        
        # Prepare Batch Tensors
        gen_imgs_cpu = torch.stack([item['gen_img'] for item in batch_items])
        gen_imgs = gen_imgs_cpu.to(device)
        
        # Load Source Images (Reloading to avoid keeping all src in RAM)
        src_tensors = []
        for item in batch_items:
            img = Image.open(item['src_path']).convert('RGB').resize((256, 256))
            src_tensors.append(T.ToTensor()(img))
        src_imgs = torch.stack(src_tensors).to(device)
        
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            # 1. Content LPIPS (Batch)
            c_lpips_vals = []
            if loss_fn:
                # LPIPS forward supports batch
                dists = loss_fn(to_lpips_input(gen_imgs), to_lpips_input(src_imgs))
                c_lpips_vals = dists.view(-1).cpu().float().numpy()
            else:
                c_lpips_vals = [0.0] * len(batch_items)

            # 2. CLIP Features (Batch)
            gen_clips = None
            src_clips = None
            c_clip_scores = [0.0] * len(batch_items)
            
            if has_clip:
                # Gen CLIP
                pil_gens = [to_pil(img.float()) for img in gen_imgs_cpu]
                inputs_gen = clip_processor(images=pil_gens, return_tensors='pt').to(device)
                gen_clips = clip_model.get_image_features(**inputs_gen)
                gen_clips = gen_clips / gen_clips.norm(p=2, dim=-1, keepdim=True) # [B, 512]
                
                # Src CLIP
                pil_srcs = [to_pil(img.cpu().float()) for img in src_imgs]
                inputs_src = clip_processor(images=pil_srcs, return_tensors='pt').to(device)
                src_clips = clip_model.get_image_features(**inputs_src)
                src_clips = src_clips / src_clips.norm(p=2, dim=-1, keepdim=True)
                
                # Content CLIP Score (Cosine Sim)
                c_clip_scores = F.cosine_similarity(gen_clips, src_clips).cpu().float().numpy()

            # 3. Style Metrics (Per Item logic due to varying targets, but vectorized CLIP)
            for i, item in enumerate(batch_items):
                tgt_id = item['tgt_style_id']
                
                # Vectorized CLIP Style Score
                s_clip_score = 0.0
                if has_clip and tgt_id in ref_clip_matrices:
                    # [1, 512] @ [512, N_ref] -> [1, N_ref]
                    ref_matrix = ref_clip_matrices[tgt_id]
                    sims = torch.matmul(gen_clips[i:i+1], ref_matrix.t())
                    s_clip_score = sims.mean().item()
                
                # LPIPS Style Score (Looping references, sampling subset)
                s_lpips_score = 0.0
                if loss_fn:
                    tgt_refs = ref_features[tgt_id]
                    # Sample subset
                    refs_sample = random.sample(tgt_refs, min(len(tgt_refs), args.max_ref_compare))
                    
                    if refs_sample:
                        ref_imgs = []
                        for r in refs_sample:
                             ref_imgs.append(T.ToTensor()(Image.open(r['path']).convert('RGB').resize((256,256))))
                        ref_batch = torch.stack(ref_imgs).to(device)
                        gen_expanded = gen_imgs[i:i+1].expand(len(ref_batch), -1, -1, -1)
                        
                        dists = loss_fn(to_lpips_input(gen_expanded), to_lpips_input(ref_batch))
                        s_lpips_score = dists.mean().item()

                # Write Row
                writer.writerow({
                    'src_style': item['src_style'],
                    'tgt_style': item['tgt_style_name'],
                    'src_image': item['src_path'].name,
                    'gen_image': item['gen_name'],
                    'content_lpips': c_lpips_vals[i],
                    'style_lpips': s_lpips_score,
                    'clip_style': s_clip_score,
                    'clip_content': c_clip_scores[i]
                })
            
            csv_file.flush()

    csv_file.close()
    io_pool.shutdown(wait=True)
    
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