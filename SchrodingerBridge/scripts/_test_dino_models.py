"""Quick DINO eval with different models to find the one matching 0.4917."""
import torch, torch.nn.functional as F, json, sys, os
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModel
import csv

DINO_TRANSFORM = T.Compose([
    T.Resize(224, interpolation=Image.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_dino(model_name, device, cache_dir):
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    # Try local first
    parts = model_name.split("/")
    if len(parts) == 2 and cache_dir:
        repo_dir = Path(cache_dir) / "hub" / f"models--{parts[0]}--{parts[1]}"
        snap_root = repo_dir / "snapshots"
        if snap_root.exists():
            revisions = [p for p in snap_root.iterdir() if p.is_dir()]
            if revisions:
                print(f"  Loading from local: {revisions[0]}")
                return AutoModel.from_pretrained(str(revisions[0])).to(device).eval()
    print(f"  Loading from HF: {model_name}")
    return AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()

def load_image(path):
    with Image.open(path) as img:
        return img.convert("RGB")

@torch.inference_mode()
def extract_features(paths, model, device, batch_size):
    cls_features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        pixels = torch.stack([DINO_TRANSFORM(load_image(p)) for p in batch_paths]).to(device)
        output = model(pixels, output_hidden_states=True)
        cls_features.append(F.normalize(output.last_hidden_state[:, 0, :].float(), dim=-1).cpu())
    return torch.cat(cls_features, dim=0)

def compute_dino_s(generated_cls, style_refs_cls):
    # style_refs_cls: (num_refs, dim), generated_cls: (num_gen, dim)
    scores = []
    for i in range(generated_cls.shape[0]):
        sims = style_refs_cls @ generated_cls[i]
        scores.append(sims.max().item())
    return float(sum(scores) / len(scores))

# Load metrics.csv to get paths
eval_dir = Path(r"C:\Users\Administrator\_tmp_bf16_eval")
test_dir = Path(r"I:\Github\Latent_Style\WEAVE\data\test")
cache_dir = r"I:\Github\Latent_Style\WEAVE\runs\cache\hf"

rows = list(csv.DictReader((eval_dir / "metrics.csv").open(encoding="utf-8-sig")))
print(f"Loaded {len(rows)} rows from metrics.csv")

# Resolve image paths
gen_paths = []
src_paths = []
for row in rows:
    gp = eval_dir / "images" / row["gen_image"]
    if not gp.exists():
        gp = eval_dir / row["gen_image"]
    sp = test_dir / row["src_style"] / row["src_image"]
    if not sp.exists():
        sp = test_dir / row["src_style"] / f"{row['src_style']}__{row['src_image']}"
    gen_paths.append(gp)
    src_paths.append(sp)

# Collect style reference images
styles = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
style_refs = {}
for style in styles:
    style_dir = test_dir / style
    refs = sorted([p for p in style_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    style_refs[style] = refs[:30]  # limit to 30 refs

# Test different models
models_to_test = [
    "facebook/dinov2-small",
    "facebook/dinov2-base",
]

device = "cuda"
for model_name in models_to_test:
    print(f"\n=== Testing {model_name} ===")
    try:
        model = load_dino(model_name, device, cache_dir)
    except Exception as e:
        print(f"  FAILED: {e}")
        continue
    
    # Extract style reference features
    all_ref_paths = []
    ref_style_map = []
    for style in styles:
        for p in style_refs[style]:
            all_ref_paths.append(p)
            ref_style_map.append(style)
    print(f"  Extracting {len(all_ref_paths)} style reference features...")
    ref_cls = extract_features(all_ref_paths, model, device, batch_size=8)
    
    # Extract generated features
    print(f"  Extracting {len(gen_paths)} generated features...")
    gen_cls = extract_features(gen_paths, model, device, batch_size=8)
    
    # Compute DINO-S per target style
    dino_s_all = []
    for i, row in enumerate(rows):
        tgt_style = row["tgt_style"]
        ref_indices = [j for j, s in enumerate(ref_style_map) if s == tgt_style]
        ref_cls_style = ref_cls[ref_indices]
        sims = ref_cls_style @ gen_cls[i]
        dino_s_all.append(sims.max().item())
    
    mean_dino_s = float(sum(dino_s_all) / len(dino_s_all))
    
    # Compute DINO-C
    src_cls = extract_features(src_paths, model, device, batch_size=8)
    dino_c_all = [float((gen_cls[i] * src_cls[i]).sum().item()) for i in range(len(rows))]
    mean_dino_c = float(sum(dino_c_all) / len(dino_c_all))
    
    print(f"  DINO-S = {mean_dino_s:.4f}")
    print(f"  DINO-C = {mean_dino_c:.4f}")
    print(f"  Target: DINO-S=0.4917, DINO-C=0.7782")
    
    del model
    torch.cuda.empty_cache()