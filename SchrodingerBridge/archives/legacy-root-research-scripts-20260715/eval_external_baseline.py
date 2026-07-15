#!/usr/bin/env python3
"""
Evaluate external baseline images (SaMAM, SaMST, etc.) using our unified metric protocol.
This script takes a directory of generated images and evaluates them with the same
CLIP-style, LPIPS, clip_s_delta_idt metrics used in our FC-SB experiments.

Usage:
  python eval_external_baseline.py --image-dir PATH --style-names Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e --output PATH

Expected image directory structure:
  image-dir/
    {tgt_style}/           # one subdir per target style
      *.png                # generated images named with source info

OR flat structure with naming pattern:
  {src_style}__{src_name}__to__{tgt_style}.png
"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import numpy as np

# Add src to path
_SRC_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# ---------- CLIP Setup ----------
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None

def get_clip_model(device='cuda'):
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    
    _CLIP_DEVICE = device
    
    # Try open_clip first (same as our main eval)
    try:
        import open_clip
        # Check for local cache
        cache_dir = Path(__file__).resolve().parents[1] / 'eval_cache' / 'manual_clip' / 'openai-clip-vit-base-patch32'
        if cache_dir.exists():
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=str(cache_dir))
        else:
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model = model.to(device).eval()
        _CLIP_MODEL = model
        _CLIP_PREPROCESS = preprocess
        print(f"Loaded open_clip ViT-B-32 on {device}")
        return model, preprocess, device
    except ImportError:
        pass
    
    # Fallback to HF transformers
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _CLIP_MODEL = model
    _CLIP_PREPROCESS = processor
    print(f"Loaded HF CLIP ViT-B-32 on {device}")
    return model, processor, device


def clip_image_features(images, model, preprocess, device, batch_size=32):
    """Extract CLIP image features. images: list of PIL Images."""
    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        try:
            import open_clip
            if isinstance(preprocess, open_clip.transform.ImageTransform):
                tensors = torch.stack([preprocess(img) for img in batch]).to(device)
                with torch.no_grad():
                    features = model.encode_image(tensors)
                    features = F.normalize(features, dim=-1)
                all_features.append(features)
                continue
        except ImportError:
            pass
        # HF transformers
        inputs = preprocess(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            # HF can return tensor or BaseModelOutputWithPooling
            if hasattr(outputs, 'image_embeds'):
                features = outputs.image_embeds
            elif isinstance(outputs, torch.Tensor):
                features = outputs
            else:
                features = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state[:, 0]
            features = F.normalize(features, dim=-1)
        all_features.append(features)
    return torch.cat(all_features, dim=0) if all_features else torch.tensor([])


def compute_clip_style(output_imgs, style_ref_features):
    """Compute CLIP-style: average cosine similarity between output and style references."""
    model, preprocess, device = get_clip_model()
    output_features = clip_image_features(output_imgs, model, preprocess, device)
    # Average similarity across all style references
    sims = output_features @ style_ref_features.T  # (n_output, n_ref)
    return sims.mean().item()


def compute_clip_content(output_imgs, content_imgs):
    """Compute CLIP-content: cosine similarity between output and content source."""
    model, preprocess, device = get_clip_model()
    output_features = clip_image_features(output_imgs, model, preprocess, device)
    content_features = clip_image_features(content_imgs, model, preprocess, device)
    sims = (output_features * content_features).sum(dim=-1)
    return sims.mean().item()


# ---------- LPIPS Setup ----------
_LPIPS_MODEL = None

def get_lpips_model(device='cuda'):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    import lpips
    _LPIPS_MODEL = lpips.LPIPS(net='vgg').to(device).eval()
    return _LPIPS_MODEL


def compute_lpips(img1, img2, device='cuda'):
    """Compute LPIPS between two PIL images."""
    model = get_lpips_model(device)
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    t1 = transform(img1).unsqueeze(0).to(device)
    t2 = transform(img2).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t1, t2).item()


def compute_batch_lpips(imgs1, imgs2, device='cuda', batch_size=8):
    """Compute LPIPS for batches of image pairs."""
    model = get_lpips_model(device)
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    results = []
    for i in range(0, len(imgs1), batch_size):
        batch1 = torch.stack([transform(img) for img in imgs1[i:i+batch_size]]).to(device)
        batch2 = torch.stack([transform(img) for img in imgs2[i:i+batch_size]]).to(device)
        with torch.no_grad():
            results.extend(model(batch1, batch2).squeeze().cpu().tolist())
    return results


# ---------- Image Loading ----------
def parse_filename(filename, style_names):
    """Parse filename to extract source/target style info.
    
    Supports patterns:
    - {src_style}__{src_name}__to__{tgt_style}.png
    - {tgt_style}/{src_name}.png  (when in subdir)
    """
    name = Path(filename).stem
    if '__to__' in name:
        parts = name.split('__to__')
        tgt_style = parts[-1]
        src_part = '__to__'.join(parts[:-1])
        src_style = None
        for s in style_names:
            if src_part.startswith(s + '__'):
                src_style = s
                src_name = src_part[len(s)+2:]
                break
        return {'src_style': src_style, 'src_name': src_name, 'tgt_style': tgt_style}
    return None


def load_images_from_dir(image_dir, style_names):
    """Load images and organize by target style."""
    image_dir = Path(image_dir)
    images_by_target = defaultdict(list)
    
    # Check if organized by subdirectories
    subdirs = [d for d in image_dir.iterdir() if d.is_dir()]
    if subdirs:
        for subdir in subdirs:
            style_name = subdir.name
            for img_path in sorted(subdir.glob('*.png')):
                info = parse_filename(img_path.name, style_names) or {}
                info['path'] = str(img_path)
                info['tgt_style'] = info.get('tgt_style', style_name)
                images_by_target[info['tgt_style']].append(info)
    else:
        # Flat structure
        for img_path in sorted(image_dir.glob('*.png')):
            info = parse_filename(img_path.name, style_names)
            if info:
                info['path'] = str(img_path)
                images_by_target[info['tgt_style']].append(info)
    
    return images_by_target


# ---------- Main Evaluation ----------
def evaluate_baseline(image_dir, style_names, content_dir, output_path, device='cuda'):
    """Evaluate baseline images using unified protocol."""
    
    # 1. Load generated images
    images_by_target = load_images_from_dir(image_dir, style_names)
    total_images = sum(len(v) for v in images_by_target.values())
    print(f"Found {total_images} images across {len(images_by_target)} target styles")
    
    if total_images == 0:
        print("ERROR: No images found!")
        return
    
    # 2. Load CLIP model
    model, preprocess, clip_device = get_clip_model(device)
    
    # 3. Load style reference images for each style
    print("Loading style reference features...")
    style_ref_features = {}
    
    # Multiple candidate paths for style references
    style_roots = []
    if content_dir:
        style_roots.append(Path(content_dir))
    # Common dataset paths
    for candidate in [
        Path(r'I:\wikiart_distinct5_samam_512_classview\test'),
        Path(r'I:\datasets\wikiart_distinct5_512_images'),
        Path(r'I:\wikiart_distinct5_latents_512_ema'),
    ]:
        if candidate.exists():
            style_roots.append(candidate)
    
    for style_name in style_names:
        ref_images = []
        for style_root in style_roots:
            # Try multiple patterns
            for pattern in [style_root / style_name, style_root / 'style' / style_name, 
                           style_root / 'test' / style_name]:
                if pattern.exists() and pattern.is_dir():
                    for ext in ['*.png', '*.jpg', '*.jpeg']:
                        ref_images.extend([Image.open(p).convert('RGB') for p in sorted(pattern.glob(ext))[:50]])
                        if ref_images:
                            break
            if ref_images:
                break
        
        if ref_images:
            features = clip_image_features(ref_images, model, preprocess, clip_device)
            style_ref_features[style_name] = features
            print(f"  {style_name}: {len(ref_images)} references")
        else:
            print(f"  Warning: no style references for {style_name}")
    
    # 4. Evaluate each target style group
    all_results = []
    all_clip_style = []
    all_lpips = []
    all_clip_content = []
    
    for tgt_style, image_infos in sorted(images_by_target.items()):
        print(f"\nEvaluating {tgt_style}: {len(image_infos)} images")
        
        # Load generated images
        gen_images = []
        for info in image_infos:
            try:
                img = Image.open(info['path']).convert('RGB')
                # Resize to 512 if needed (match our eval protocol)
                if max(img.size) != 512:
                    img = img.resize((512, 512), Image.LANCZOS)
                gen_images.append(img)
            except Exception as e:
                print(f"  Warning: failed to load {info['path']}: {e}")
        
        if not gen_images:
            continue
        
        # Compute CLIP-style
        if tgt_style in style_ref_features:
            gen_features = clip_image_features(gen_images, model, preprocess, clip_device)
            clip_style_vals = (gen_features @ style_ref_features[tgt_style].T).mean(dim=1).cpu().tolist()
            avg_clip_style = sum(clip_style_vals) / len(clip_style_vals)
        else:
            clip_style_vals = [0.0] * len(gen_images)
            avg_clip_style = 0.0
            print(f"  Warning: no style references for {tgt_style}")
        
        # Compute LPIPS (against content source if available)
        lpips_vals = [0.0] * len(gen_images)  # Placeholder - need content sources
        avg_lpips = 0.0
        
        print(f"  clip_style={avg_clip_style:.4f}")
        
        for i, info in enumerate(image_infos[:len(gen_images)]):
            result = {
                'tgt_style': tgt_style,
                'src_style': info.get('src_style', ''),
                'src_name': info.get('src_name', ''),
                'clip_style': clip_style_vals[i] if i < len(clip_style_vals) else 0.0,
                'content_lpips': lpips_vals[i],
            }
            all_results.append(result)
            all_clip_style.append(result['clip_style'])
            all_lpips_vals = result['content_lpips']
    
    # 5. Compute overall summary
    overall_clip_style = sum(all_clip_style) / len(all_clip_style) if all_clip_style else 0
    overall_lpips = sum(r['content_lpips'] for r in all_results) / len(all_results) if all_results else 0
    
    # IDT floor for reference
    idt_floor = 0.6399
    clip_s_delta_idt = overall_clip_style - idt_floor
    
    summary = {
        'analysis': {
            'all_pairs_overview': {
                'clip_style': overall_clip_style,
                'content_lpips': overall_lpips,
                'clip_s_delta_idt': clip_s_delta_idt,
                'one_minus_lpips': round(1 - overall_lpips, 4) if overall_lpips > 0 else None,
                'n_images': len(all_results),
            }
        },
        'per_pair_results': all_results,
    }
    
    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    with open(output_path / 'metrics.csv', 'w', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=['tgt_style', 'src_style', 'src_name', 'clip_style', 'content_lpips'])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"  Total images: {len(all_results)}")
    print(f"  CLIP-style:   {overall_clip_style:.4f}")
    print(f"  LPIPS:        {overall_lpips:.4f}")
    print(f"  1-LPIPS:      {1-overall_lpips:.4f}" if overall_lpips > 0 else "")
    print(f"  delta_idt:    {clip_s_delta_idt:+.4f}")
    print(f"  Saved to:     {output_path}")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate external baseline images')
    parser.add_argument('--image-dir', required=True, help='Directory containing generated images')
    parser.add_argument('--style-names', default='Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e')
    parser.add_argument('--content-dir', default=None, help='Content/reference image root')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    style_names = [s.strip() for s in args.style_names.split(',')]
    evaluate_baseline(args.image_dir, style_names, args.content_dir, args.output, args.device)
