"""Measure S2WAT inference timing for N images, extrapolate to 750"""
import os, time, torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from model.configuration import TransModule_Config
from model.s2wat import S2WAT
from net import TransModule, Decoder_MVGG
from tools import Sample_Test_Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Build model
trans_cfg = TransModule_Config(nlayer=3, d_model=768, nhead=8, mlp_ratio=4,
    qkv_bias=False, attn_drop=0., drop=0., drop_path=0.,
    act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=True)
encoder = S2WAT(img_size=256, patch_size=2, in_chans=3, embed_dim=192,
    depths=[2,2,2], nhead=[3,6,12], strip_width=[2,4,7],
    drop_path_rate=0., patch_norm=True)
decoder = Decoder_MVGG(d_model=768, seq_input=True)
trans = TransModule(trans_cfg)
net = Sample_Test_Net(encoder, decoder, trans).to(device)
net.eval()

print(f"Model loaded on {device} | params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")

# Load test images
test_root = Path("../../../style_data/overfit50")
tfm = T.Compose([T.Resize((256,256)), T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
untransform = T.Compose([T.Normalize(mean=[0,0,0], std=[1/0.229,1/0.224,1/0.225]),
    T.Normalize(mean=[-0.485,-0.456,-0.406], std=[1,1,1])])

styles = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
images = {}
for s in styles:
    files = sorted((test_root / s).glob("*"))
    images[s] = [tfm(Image.open(f).convert("RGB")) for f in files[:6]]

count = 0
total_time = 0.0
with torch.no_grad():
    for src_s in styles:
        for src_img in images[src_s]:
            for tgt_s in styles:
                tgt_img = images[tgt_s][0]
                src_b = src_img.unsqueeze(0).to(device)
                tgt_b = tgt_img.unsqueeze(0).to(device)
                
                t0 = time.perf_counter()
                out = net(src_b, tgt_b)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                total_time += (t1 - t0)
                count += 1

avg = total_time / count
extrap_750 = avg * 750
print(f"\nInference: {count} images in {total_time:.3f}s")
print(f"Average: {avg*1000:.2f} ms/image")
print(f"Extrapolated to 750 images: {extrap_750:.1f}s ({extrap_750/60:.1f} min)")
