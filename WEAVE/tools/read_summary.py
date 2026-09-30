#!/usr/bin/env python3
import json

with open("/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_nswd_gate03_smoke/full_eval/epoch_0001/summary.json") as f:
    s = json.load(f)

r = s.get("runtime_observability", {}).get("all_pairs_overview", {})
an = s.get("analysis", {}).get("all_pairs_overview", {})
st = s.get("analysis", {}).get("style_transfer_ability", {})
idr = s.get("analysis", {}).get("identity_reconstruction", {})

print("=" * 50)
print("GATE_VALUE:", r.get("model_style_gate_value"))
print("VELOCITY_ABS:", r.get("model_velocity_abs"))
print("ENDPOINT_PRED_ABS:", r.get("model_endpoint_pred_abs"))
print("ENDPOINT_HIGH_ABS:", r.get("model_endpoint_high_abs"))
print("ENDPOINT_LOW_ABS:", r.get("model_endpoint_low_abs"))
print("=" * 50)
print("ALL_PAIRS CLIP_STYLE:", an.get("clip_style"))
print("ALL_PAIRS CLIP_S_DELTA_IDT:", an.get("clip_s_delta_idt"))
print("ALL_PAIRS CONTENT_LPIPS:", an.get("content_lpips"))
print("=" * 50)
print("STYLE_TRANSFER CLIP_STYLE:", st.get("clip_style"))
print("STYLE_TRANSFER CLIP_S_DELTA_IDT:", st.get("clip_s_delta_idt"))
print("IDENTITY CLIP_STYLE:", idr.get("clip_style"))
print("=" * 50)