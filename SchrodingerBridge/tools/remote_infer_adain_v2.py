"""AdaIN inference on distinct5_512 - run on remote server."""
import os, sys, torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.models as models

# ── Config ──
STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']
TEST_DIR = Path(r'I:\wikiart_distinct5_samam_512_classview\test')
OUT_DIR = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\adain_v32k')
VGG_PATH = Path(r'I:\GitHub\Latent_Style\Related_Works\repos\SaMam\LOSS\vgg_ckp\vgg_normalised.pth')
DECODER_PATH = Path(r'I:\GitHub\Latent_Style\Related_Works\repos\pytorch-AdaIN\models\decoder.pth')
SIZE = 512

# VGG encoder (up to relu4_1)
def build_vgg_encoder(vgg_path):
    vgg = models.vgg.vgg19(weights=None)
    if vgg_path and vgg_path.exists():
        state = torch.load(vgg_path, map_location='cpu', weights_only=True)
        vgg.load_state_dict(state)
    enc_layers = list(vgg.features)[:22]  # up to relu4_1
    return nn.Sequential(*enc_layers)

# AdaIN
def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean = style_feat.mean(dim=[2,3], keepdim=True)
    style_std = style_feat.std(dim=[2,3], keepdim=True) + 1e-8
    content_mean = content_feat.mean(dim=[2,3], keepdim=True)
    content_std = content_feat.std(dim=[2,3], keepdim=True) + 1e-8
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean

def run_adain():
    print("Loading VGG encoder...")
    encoder = build_vgg_encoder(VGG_PATH)
    encoder.eval()

    # Check decoder
    if not DECODER_PATH.exists():
        # Try to download
        print(f"decoder.pth not found at {DECODER_PATH}")
        print("Downloading from GitHub release...")
        import urllib.request
        DECODER_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"
        urllib.request.urlretrieve(url, str(DECODER_PATH))
        print(f"Downloaded to {DECODER_PATH}")

    print("Loading decoder...")
    decoder_state = torch.load(str(DECODER_PATH), map_location='cpu', weights_only=True)
    
    # Build decoder - use pytorch-AdaIN repo if available
    adain_repo = Path(r'I:\GitHub\Latent_Style\Related_Works\repos\pytorch-AdaIN')
    if adain_repo.exists():
        sys.path.insert(0, str(adain_repo))
        from decoder import decoder as AdaINDecoder
        decoder = AdaINDecoder()
        decoder.load_state_dict(decoder_state)
    else:
        print("ERROR: pytorch-AdaIN repo not found, cannot build decoder")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    decoder = decoder.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
    ])

    # Collect test images
    items = []
    for style in STYLES:
        style_dir = TEST_DIR / style
        for f in sorted(style_dir.iterdir()):
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                items.append((style, f.stem, str(f)))

    # Collect style references (first image per style)
    style_refs = {}
    for style in STYLES:
        style_dir = TEST_DIR / style
        for f in sorted(style_dir.iterdir()):
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                style_refs[style] = str(f)
                break

    print(f"Found {len(items)} test images, {len(style_refs)} style refs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done_marker = OUT_DIR / '_DONE'
    if done_marker.exists():
        print("Already done, skipping")
        return

    count = 0
    with torch.no_grad():
        # Pre-encode style references
        style_features = {}
        for tgt_style, ref_path in style_refs.items():
            ref_img = Image.open(ref_path).convert('RGB')
            ref_tensor = transform(ref_img).unsqueeze(0).to(device)
            style_features[tgt_style] = encoder(ref_tensor)

        # Process each source image
        for src_style, src_stem, src_path in items:
            src_img = Image.open(src_path).convert('RGB')
            src_tensor = transform(src_img).unsqueeze(0).to(device)
            content_feat = encoder(src_tensor)

            for tgt_style in STYLES:
                out_name = f'{src_style}__{src_stem}__to__{tgt_style}.png'
                out_path = OUT_DIR / out_name
                if out_path.exists():
                    count += 1
                    continue

                t = adaptive_instance_normalization(content_feat, style_features[tgt_style])
                out_feat = decoder(t)
                out_img = out_feat.cpu().squeeze(0)
                # Denormalize
                out_img = out_img.clamp(0, 1)
                pil_img = transforms.ToPILImage()(out_img)
                pil_img.save(str(out_path))
                count += 1

            if count % 50 == 0:
                print(f"  {count}/750 done")

    done_marker.write_text(f'{count} images')
    print(f"AdaIN Complete: {count} images")

if __name__ == '__main__':
    print("=" * 60)
    print("AdaIN Inference on distinct5_512")
    print("=" * 60)
    run_adain()
    print("ALL DONE")
