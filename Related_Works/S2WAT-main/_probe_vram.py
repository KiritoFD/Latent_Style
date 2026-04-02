import json
import torch
import torch.nn as nn
from model.configuration import TransModule_Config
from model.s2wat import S2WAT
from net import vgg, TransModule, Decoder_MVGG, Net

vgg_path = r"G:/GitHub/Latent_Style/Related_Works/S2WAT-main/pre_trained_models/vgg_normalised.pth"

if not torch.cuda.is_available():
    raise SystemExit('CUDA not available')

device = torch.device('cuda')

# Build model
trans_cfg = TransModule_Config(
    nlayer=3, d_model=768, nhead=8, mlp_ratio=4,
    qkv_bias=False, attn_drop=0., drop=0., drop_path=0.,
    act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=True
)

vgg.load_state_dict(torch.load(vgg_path, map_location='cpu'))
encoder = S2WAT(img_size=224, patch_size=2, in_chans=3, embed_dim=192, depths=[2,2,2], nhead=[3,6,12], strip_width=[2,4,7], drop_path_rate=0., patch_norm=True)
decoder = Decoder_MVGG(d_model=768, seq_input=True)
trans = TransModule(trans_cfg)
net = Net(encoder, decoder, trans, vgg).to(device)
opt = torch.optim.Adam([
    {'params': net.encoder.parameters()},
    {'params': net.decoder.parameters()},
    {'params': net.transModule.parameters()},
], lr=1e-4)

results = []
for bs in range(1, 17):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    ok = True
    err = ''
    peak_alloc = peak_reserved = None
    try:
        ic = torch.randn(bs, 3, 224, 224, device=device)
        isty = torch.randn(bs, 3, 224, 224, device=device)
        loss_c, loss_s, loss_id1, loss_id2, _ = net(ic, isty)
        loss = 2*loss_c + 3*loss_s + 50*loss_id1 + 1*loss_id2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024 / 1024
    except RuntimeError as e:
        ok = False
        err = str(e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results.append({
        'batch_size': bs,
        'ok': ok,
        'peak_alloc_mb': peak_alloc,
        'peak_reserved_mb': peak_reserved,
        'err': err[:200]
    })
    print(results[-1])
    if not ok and ('out of memory' in err.lower()):
        break

print('RESULT_JSON_START')
print(json.dumps(results, ensure_ascii=False, indent=2))
print('RESULT_JSON_END')
