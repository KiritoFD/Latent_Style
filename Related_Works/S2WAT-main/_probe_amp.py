import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from model.configuration import TransModule_Config
from model.s2wat import S2WAT
from net import vgg, TransModule, Decoder_MVGG, Net
vgg_path=r"G:/GitHub/Latent_Style/Related_Works/S2WAT-main/pre_trained_models/vgg_normalised.pth"

device='cuda'
trans_cfg=TransModule_Config(nlayer=3,d_model=768,nhead=8,mlp_ratio=4,qkv_bias=False,attn_drop=0.,drop=0.,drop_path=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm,norm_first=True)
vgg.load_state_dict(torch.load(vgg_path,map_location='cpu'))
enc=S2WAT(img_size=224,patch_size=2,in_chans=3,embed_dim=192,depths=[2,2,2],nhead=[3,6,12],strip_width=[2,4,7],drop_path_rate=0.,patch_norm=True)
dec=Decoder_MVGG(d_model=768,seq_input=True)
trans=TransModule(trans_cfg)
net=Net(enc,dec,trans,vgg).to(device)
opt=torch.optim.Adam([{'params':net.encoder.parameters()},{'params':net.decoder.parameters()},{'params':net.transModule.parameters()}],lr=1e-4)
scaler=GradScaler('cuda')

for mode in ['fp32','amp']:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ic=torch.randn(1,3,256,256,device=device)
    isty=torch.randn(1,3,256,256,device=device)
    opt.zero_grad(set_to_none=True)
    if mode=='amp':
        with autocast('cuda', dtype=torch.float16):
            lc,ls,li1,li2,_=net(ic,isty)
            loss=2*lc+3*ls+50*li1+li2
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    else:
        lc,ls,li1,li2,_=net(ic,isty)
        loss=2*lc+3*ls+50*li1+li2
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(mode, torch.cuda.max_memory_reserved()/1024/1024)
