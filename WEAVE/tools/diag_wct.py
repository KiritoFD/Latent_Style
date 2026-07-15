"""Diagnose WCT: check if style refs are different and if WCT produces style-dependent output."""
import sys
import hashlib
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from infer_wct import create_model, wct, TEST_DIR, STYLE_NAMES, IMAGE_SIZE

device = torch.device("cuda")
print("=== Style Reference Check ===")
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

style_refs = {}
for style_name in STYLE_NAMES:
    style_dir = TEST_DIR / style_name
    imgs = sorted(style_dir.glob("*.jpg"))
    if imgs:
        ref_img = Image.open(imgs[0]).convert("RGB")
        ref_tensor = transform(ref_img).unsqueeze(0).to(device)
        style_refs[style_name] = ref_tensor
        # Hash the raw image bytes
        img_hash = hashlib.md5(imgs[0].read_bytes()).hexdigest()[:8]
        # Hash the tensor
        tensor_hash = hashlib.md5(ref_tensor.cpu().numpy().tobytes()).hexdigest()[:8]
        print(f"  {style_name}: file={imgs[0].name} file_hash={img_hash} tensor_hash={tensor_hash}")

print("\n=== WCT Output Check ===")
model = create_model("wct_v32k", device, alpha=1.0)

# Load one content image
content_dir = TEST_DIR / "Early_Renaissance"
content_imgs = sorted(content_dir.glob("*.jpg"))
content_img = Image.open(content_imgs[0]).convert("RGB")
content_tensor = transform(content_img).unsqueeze(0).to(device)

# Apply WCT with different style references
print("Content image:", content_imgs[0].name)
for tgt_style in STYLE_NAMES:
    if tgt_style not in style_refs:
        continue
    style_ref = style_refs[tgt_style]
    output = model.transfer(content_tensor, style_ref)
    out_hash = hashlib.md5(output.cpu().numpy().tobytes()).hexdigest()[:8]
    out_min = output.min().item()
    out_max = output.max().item()
    out_mean = output.mean().item()
    print(f"  -> {tgt_style}: hash={out_hash} min={out_min:.4f} max={out_max:.4f} mean={out_mean:.4f}")

print("\n=== Feature Check ===")
# Check if content features are the same
with torch.no_grad():
    c_feats = model.encoder(content_tensor)
    print(f"Content feat[-1] shape: {c_feats[-1].shape}")
    print(f"Content feat[-1] mean: {c_feats[-1].mean().item():.4f} std: {c_feats[-1].std().item():.4f}")

    for tgt_style in STYLE_NAMES:
        if tgt_style not in style_refs:
            continue
        s_feats = model.encoder(style_refs[tgt_style])
        s_mean = s_feats[-1].mean().item()
        s_std = s_feats[-1].std().item()
        print(f"  Style {tgt_style} feat[-1]: mean={s_mean:.4f} std={s_std:.4f}")

        # Apply WCT manually
        t = wct(c_feats[-1], s_feats[-1], alpha=1.0)
        t_hash = hashlib.md5(t.cpu().numpy().tobytes()).hexdigest()[:8]
        t_mean = t.mean().item()
        t_std = t.std().item()
        print(f"    WCT output: hash={t_hash} mean={t_mean:.4f} std={t_std:.4f}")

print("\n==DIAG_DONE==")
