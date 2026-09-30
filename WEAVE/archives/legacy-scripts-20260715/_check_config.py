"""Check effective config parameters from saved experiment config."""
import json, os, sys

ROOT = r"I:\Github\Latent_Style\SchrodingerBridge"
exp = sys.argv[1] if len(sys.argv) > 1 else "hp_simple_swd12_15ep"

config_path = os.path.join(ROOT, "exp", exp, "config.json")
if not os.path.exists(config_path):
    print(f"Config not found: {config_path}")
    sys.exit(1)

with open(config_path) as f:
    c = json.load(f)

m = c.get("model", {})
b = c.get("bridge", {})
t = c.get("training", {})

print(f"=== {exp} effective config ===")
print(f"num_res_blocks:       {m.get('num_res_blocks', 'default')}")
print(f"num_hires_blocks:     {m.get('num_hires_blocks', 'default')}")
print(f"num_decoder_blocks:   {m.get('num_decoder_blocks', 'default')}")
print(f"base_dim:             {m.get('base_dim', 'default')}")
print(f"style_dim:            {m.get('style_dim', 'default')}")
print(f"style_attn_num_tokens:{m.get('style_attn_num_tokens', 'default')}")
print(f"style_attn_num_heads: {m.get('style_attn_num_heads', 'default')}")
print(f"style_cross_attn_gate_init: {m.get('style_cross_attn_gate_init', 'default')}")
print(f"style_attn_sharpen_scale:   {m.get('style_attn_sharpen_scale', 'default')}")
print(f"style_attn_temperature:     {m.get('style_attn_temperature', 'default')}")
print(f"dwt_route_train_prob: {m.get('dwt_route_train_prob', 'default')}")
print(f"cross_attn_dwt_route: {m.get('cross_attn_dwt_route', 'default')}")
print(f"style_film_heads:     {m.get('style_film_heads', 'default')}")
print(f"enable_hh_head:       {m.get('enable_hh_head', 'default')}")
print(f"endpoint_adain_mode:  {m.get('endpoint_adain_mode', 'default')}")
print(f"endpoint_adain_scale: {m.get('endpoint_adain_scale', 'default')}")
print(f"style_extrap_alpha:   {m.get('style_extrap_alpha', 'default')}")
print(f"---")
print(f"single_step_swd_weight: {b.get('single_step_swd_weight', 'default')}")
print(f"swd_scale_mode:       {b.get('swd_scale_mode', 'default')}")
print(f"swd_distance_mode:    {b.get('swd_distance_mode', 'default')}")
print(f"swd_semantic_mode:    {b.get('swd_semantic_mode', 'default')}")
print(f"spectral_ode_enabled: {b.get('spectral_ode_enabled', 'default')}")
print(f"spectral_w_ll:        {b.get('spectral_w_ll', 'default')}")
print(f"---")
print(f"batch_size:           {t.get('batch_size', 'default')}")
print(f"num_epochs:           {t.get('num_epochs', 'default')}")
