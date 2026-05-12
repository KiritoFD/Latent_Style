import argparse
import json
import torch
import torch.nn as nn
from model.configuration import TransModule_Config
from model.s2wat import S2WAT
from net import vgg, TransModule, Decoder_MVGG, Net

p = argparse.ArgumentParser()
p.add_argument('--bs', type=int, required=True)
p.add_argument('--img_size', type=int, default=256)
p.add_argument('--vgg', type=str, required=True)
args = p.parse_args()

res = {'batch_size': args.bs, 'img_size': args.img_size, 'ok': False, 'peak_alloc_mb': None, 'peak_reserved_mb': None, 'err': ''}

try:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    device = torch.device('cuda')

    trans_cfg = TransModule_Config(
        nlayer=3, d_model=768, nhead=8, mlp_ratio=4,
        qkv_bias=False, attn_drop=0., drop=0., drop_path=0.,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=True
    )
    vgg.load_state_dict(torch.load(args.vgg, map_location='cpu'))
    encoder = S2WAT(img_size=args.img_size, patch_size=2, in_chans=3, embed_dim=192, depths=[2,2,2], nhead=[3,6,12], strip_width=[2,4,7], drop_path_rate=0., patch_norm=True)
    decoder = Decoder_MVGG(d_model=768, seq_input=True)
    trans = TransModule(trans_cfg)
    net = Net(encoder, decoder, trans, vgg).to(device)
    opt = torch.optim.Adam([
        {'params': net.encoder.parameters()},
        {'params': net.decoder.parameters()},
        {'params': net.transModule.parameters()},
    ], lr=1e-4)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    ic = torch.randn(args.bs, 3, args.img_size, args.img_size, device=device)
    isty = torch.randn(args.bs, 3, args.img_size, args.img_size, device=device)
    loss_c, loss_s, loss_id1, loss_id2, _ = net(ic, isty)
    loss = 2*loss_c + 3*loss_s + 50*loss_id1 + loss_id2
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()

    res['ok'] = True
    res['peak_alloc_mb'] = float(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
    res['peak_reserved_mb'] = float(torch.cuda.max_memory_reserved(device) / 1024 / 1024)
except Exception as e:
    res['err'] = str(e)

print(json.dumps(res, ensure_ascii=False))
