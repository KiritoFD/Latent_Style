import torch
c = torch.load("G:/GitHub/Latent_Style/eval_cache/offline_pairing/dinov2_wikiart_distinct5_samam_512_train_cache.pt", map_location="cpu", weights_only=True)
if isinstance(c, dict):
    for k, v in list(c.items())[:3]:
        if hasattr(v, "shape"):
            print(k, v.shape, v.dtype)
        else:
            print(k, type(v).__name__)
else:
    print(type(c).__name__)
