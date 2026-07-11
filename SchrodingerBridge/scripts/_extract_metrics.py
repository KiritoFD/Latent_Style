"""Extract metrics from summary.json and config.json for Latent-WCT and T1 ASG."""
import json
import sys
from pathlib import Path

# Latent-WCT summary
wct_summary_path = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\full_eval\epoch_0000\summary.json")
if wct_summary_path.exists():
    s = json.load(open(wct_summary_path))
    a = s.get("analysis", {})
    t = a.get("style_transfer_ability", {})
    ap = a.get("all_pairs_overview", {})
    print("=== Latent-WCT Results ===")
    print(f"CLIP-S (transfer): {t.get('clip_style', 'N/A')}")
    print(f"LPIPS (transfer):  {t.get('content_lpips', 'N/A')}")
    print(f"CLIP-S (allpairs): {ap.get('clip_style', 'N/A')}")
    print(f"LPIPS (allpairs):  {ap.get('content_lpips', 'N/A')}")
    print(f"Generated count:   {s.get('generated_count', 'N/A')}")
else:
    print("WCT summary not found!")

# DINO results
dino_path = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\_dino_results\latent_wct.json")
if dino_path.exists():
    d = json.load(open(dino_path))
    print(f"\nDINO-con: {d.get('dino_content', 'N/A')}")
    print(f"DINO-sty: {d.get('dino_style', 'N/A')}")
else:
    print("\nDINO results not found!")

# T1 ASG config
print("\n=== T1 ASG Config ===")
cfg_path = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\config.json")
if cfg_path.exists():
    c = json.load(open(cfg_path))
    m = c.get("model", {})
    print(f"adaptive_style_gate: {m.get('adaptive_style_gate')}")
    print(f"lowpass_mode: {m.get('lowpass_mode')}")
    print(f"spectral_ode_enabled: {m.get('spectral_ode_enabled')}")
    print(f"endpoint_adain_mode: {m.get('endpoint_adain_mode')}")
    print(f"cross_attn_dwt_route: {m.get('cross_attn_dwt_route')}")
    print(f"dwt_route_train_prob: {m.get('dwt_route_train_prob')}")
    print(f"style_condition_source: {m.get('style_condition_source')}")
    print(f"contract_family: {m.get('contract_family')}")
    print(f"per_subband_gate: {m.get('per_subband_gate')}")
    print(f"style_gate_mode: {m.get('style_gate_mode')}")
    print(f"style_attn_mode: {m.get('style_attn_mode')}")
    fe = c.get("full_eval", {})
    print(f"full_eval.num_steps: {fe.get('num_steps')}")
    b = c.get("bridge", {})
    print(f"single_step_swd_weight: {b.get('single_step_swd_weight')}")
    print(f"terminal_swd_weight: {b.get('terminal_swd_weight')}")
    print(f"bridge_path_mode: {b.get('bridge_path_mode')}")
    print(f"training_target_projection_mode: {b.get('training_target_projection_mode')}")
else:
    print("Config not found!")

# T1 ASG existing eval results
print("\n=== T1 ASG Existing Eval ===")
t1_summary = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\full_eval\epoch_0005\summary.json")
if t1_summary.exists():
    s = json.load(open(t1_summary))
    a = s.get("analysis", {})
    t = a.get("style_transfer_ability", {})
    ap = a.get("all_pairs_overview", {})
    print(f"CLIP-S (transfer): {t.get('clip_style', 'N/A')}")
    print(f"LPIPS (transfer):  {t.get('content_lpips', 'N/A')}")
    print(f"CLIP-S (allpairs): {ap.get('clip_style', 'N/A')}")
    print(f"LPIPS (allpairs):  {ap.get('content_lpips', 'N/A')}")
    print(f"Generated count:   {s.get('generated_count', 'N/A')}")
else:
    print("T1 ASG summary not found!")

# T1 ASG DINO results
t1_dino_paths = [
    Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\_dino_results\t1_asg_5ep.json"),
    Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\_dino_results\t1_asg.json"),
]
for p in t1_dino_paths:
    if p.exists():
        d = json.load(open(p))
        print(f"\nDINO-con: {d.get('dino_content', 'N/A')}")
        print(f"DINO-sty: {d.get('dino_style', 'N/A')}")
        break
else:
    print("\nT1 ASG DINO results not found in expected locations")
    # Search for it
    dino_dir = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\_dino_results")
    if dino_dir.exists():
        print("Available DINO results:")
        for f in sorted(dino_dir.glob("*.json")):
            print(f"  {f.name}")
