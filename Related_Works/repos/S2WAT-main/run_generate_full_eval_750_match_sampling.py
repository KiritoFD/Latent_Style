import random
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.utils import save_image

from model.configuration import TransModule_Config
from model.s2wat import S2WAT
from net import TransModule, Decoder_MVGG
from tools import content_style_transTo_pt, Sample_Test_Net

ckpt = Path(r"G:/GitHub/Latent_Style/Related_Works/S2WAT-main/pre_trained_models/checkpoint_bs1_safe/checkpoint_2000_epoch.pkl")
styles_root = Path(r"G:/GitHub/Latent_Style/style_data/train")
style_names = ["photo", "cezanne", "Hayao", "monet", "vangogh"]
out_dir = Path(r"G:/GitHub/Latent_Style/Related_Works/runs/s2wat_bs1_safe_e2000_full_eval")
img_dir = out_dir / "images"
img_dir.mkdir(parents=True, exist_ok=True)

sampled_by_style = {}
for s in style_names:
    files = sorted([p for p in (styles_root / s).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])
    rng = random.Random(42)
    sampled = files[:]
    rng.shuffle(sampled)
    sampled = sampled[:30]
    sampled_by_style[s] = sampled

trans_cfg = TransModule_Config(
    nlayer=3, d_model=768, nhead=8, mlp_ratio=4,
    qkv_bias=False, attn_drop=0., drop=0., drop_path=0.,
    act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=True
)
encoder = S2WAT(
    img_size=256, patch_size=2, in_chans=3, embed_dim=192,
    depths=[2,2,2], nhead=[3,6,12], strip_width=[2,4,7],
    drop_path_rate=0., patch_norm=True
)
decoder = Decoder_MVGG(d_model=768, seq_input=True)
trans = TransModule(trans_cfg)
net = Sample_Test_Net(encoder, decoder, trans)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(str(ckpt), map_location=device, weights_only=False)
net.encoder.load_state_dict(checkpoint['encoder'])
net.decoder.load_state_dict(checkpoint['decoder'])
net.transModule.load_state_dict(checkpoint['transModule'])
net.to(device)
net.eval()

count_new = 0
count_all = 0
with torch.no_grad():
    for src_style in style_names:
        for src_path in sampled_by_style[src_style]:
            src_stem = src_path.stem
            for tgt_style in style_names:
                out_name = f"{src_style}_{src_stem}_to_{tgt_style}.jpg"
                out_path = img_dir / out_name
                count_all += 1
                if out_path.exists():
                    continue
                tgt_path = sampled_by_style[tgt_style][0]
                i_c, i_s = content_style_transTo_pt(str(src_path), str(tgt_path))
                out = net(i_c.to(device), i_s.to(device), arbitrary_input=True)
                save_image(out.cpu(), str(out_path))
                count_new += 1
                if count_new % 50 == 0:
                    print(f"new_generated {count_new}")

print(f"expected_total={count_all}")
print(f"new_generated_total={count_new}")
print(f"dir_total={len(list(img_dir.glob('*_to_*.jpg')))}")
