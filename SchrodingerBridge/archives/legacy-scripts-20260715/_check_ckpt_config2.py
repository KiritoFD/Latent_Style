import torch
ckpt = torch.load(r"I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\epoch_0005.pt", map_location="cpu", weights_only=False)
config = ckpt.get("config", {})
model_cfg = config.get("model", {})
print(f"contract_family = {model_cfg.get('contract_family', 'NOT_SET')}")
print(f"tokenizer_family = {model_cfg.get('tokenizer_family', 'NOT_SET')}")
print(f"adaptive_style_gate = {model_cfg.get('adaptive_style_gate', 'NOT_SET')}")
print(f"dwt_route = {model_cfg.get('dwt_route', 'NOT_SET')}")
print(f"dwt_route_train_prob = {model_cfg.get('dwt_route_train_prob', 'NOT_SET')}")
print(f"dwt_ll_route_alpha = {model_cfg.get('dwt_ll_route_alpha', 'NOT_SET')}")
# Print all model keys that contain 'style' or 'gate'
for k in sorted(model_cfg.keys()):
    if 'style' in k.lower() or 'gate' in k.lower() or 'asg' in k.lower():
        print(f"  {k} = {model_cfg[k]}")
