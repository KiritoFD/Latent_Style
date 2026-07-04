"""Diagnose WCT with VGG-19 encoder (non-normalised) to check if feature statistics differ."""
import sys
import hashlib
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from infer_wct import create_model, wct, TEST_DIR, STYLE_NAMES, IMAGE_SIZE

device = torch.device("cuda")
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

print("=== VGG-19 Encoder Feature Check ===")
model = create_model("wct_vgg19", device, alpha=1.0)

# Load content image
content_dir = TEST_DIR / "Early_Renaissance"
content_imgs = sorted(content_dir.glob("*.jpg"))
content_img = Image.open(content_imgs[0]).convert("RGB")
content_tensor = transform(content_img).unsqueeze(0).to(device)

# Load style references
style_refs = {}
for style_name in STYLE_NAMES:
    style_dir = TEST_DIR / style_name
    imgs = sorted(style_dir.glob("*.jpg"))
    if imgs:
        ref_img = Image.open(imgs[0]).convert("RGB")
        style_refs[style_name] = transform(ref_img).unsqueeze(0).to(device)

with torch.no_grad():
    c_feats = model.encoder(content_tensor)
    print(f"Content feat[-1] shape: {c_feats[-1].shape}")
    print(f"Content feat[-1]: min={c_feats[-1].min():.4f} max={c_feats[-1].max():.4f} mean={c_feats[-1].mean():.4f} std={c_feats[-1].std():.4f}")

    print("\nStyle features:")
    for tgt_style in STYLE_NAMES:
        if tgt_style not in style_refs:
            continue
        s_feats = model.encoder(style_refs[tgt_style])
        print(f"  {tgt_style}: min={s_feats[-1].min():.4f} max={s_feats[-1].max():.4f} mean={s_feats[-1].mean():.4f} std={s_feats[-1].std():.4f}")

        # Apply WCT
        t = wct(c_feats[-1], s_feats[-1], alpha=1.0)
        t_hash = hashlib.md5(t.cpu().numpy().tobytes()).hexdigest()[:8]
        print(f"    WCT output: hash={t_hash} min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f} std={t.std():.4f}")

        # Apply adain_post
        from infer_adain import adaptive_instance_norm
        t_post = adaptive_instance_norm(t, s_feats[-1])
        t_post_hash = hashlib.md5(t_post.cpu().numpy().tobytes()).hexdigest()[:8]
        print(f"    WCT+AdaIN_post: hash={t_post_hash} min={t_post.min():.4f} max={t_post.max():.4f} mean={t_post.mean():.4f} std={t_post.std():.4f}")

        # Decoder output
        dec_out = model.decoder(t_post)
        dec_hash = hashlib.md5(dec_out.cpu().numpy().tobytes()).hexdigest()[:8]
        print(f"    Decoder output: hash={dec_hash} min={dec_out.min():.4f} max={dec_out.max():.4f} mean={dec_out.mean():.4f}")

print("\n==DIAG_DONE==")
