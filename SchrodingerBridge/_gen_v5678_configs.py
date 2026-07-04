"""Generate V6/V7/V8 configs based on V2 (best tuning point)."""
import json
from pathlib import Path

BASE = Path(__file__).parent / "configs" / "620_spectral_v2_weights.json"
with open(BASE, "r", encoding="utf-8") as f:
    base = json.load(f)

# V6: w_ll=0.5 (more aggressive lowfreq relax), w_hh=1.5, 8ep
v6 = json.loads(json.dumps(base))
v6["bridge"]["spectral_w_ll"] = 0.5
v6["checkpoint"]["save_dir"] = "./exp/620_spectral_v6_ll05"
v6["ablation"]["name"] = "620_spectral_v6_ll05"
v6["ablation"]["axis"] = "620_spectral_ode_v6_ll05"
v6["ablation"]["notes"] = "B2 V6: w_ll=0.5 (more aggressive lowfreq relax than V2's 0.3), w_hh=1.5. Test if stronger lowfreq unlock pushes CLIP higher at cost of LPIPS."
with open(Path(__file__).parent / "configs" / "620_spectral_v6_ll05.json", "w", encoding="utf-8") as f:
    json.dump(v6, f, indent=2, ensure_ascii=False)
print("V6 written: w_ll=0.5")

# V7: V2 + V3 combo — w_ll=0.3, w_hh=1.5 + Brownian sigma=0.1, 8ep
v7 = json.loads(json.dumps(base))
v7["bridge"]["spectral_w_ll"] = 0.3
v7["bridge"]["spectral_w_hh"] = 1.5
v7["bridge"]["spectral_brownian_enabled"] = True
v7["bridge"]["spectral_brownian_sigma"] = 0.1
v7["checkpoint"]["save_dir"] = "./exp/620_spectral_v7_combo_brownian"
v7["ablation"]["name"] = "620_spectral_v7_combo_brownian"
v7["ablation"]["axis"] = "620_spectral_ode_v7_combo_brownian"
v7["ablation"]["notes"] = "B2 V7: V2+V3 combo. w_ll=0.3, w_hh=1.5 (V2 best weights) + Brownian bridge sigma=0.1 (V3 SB noise). Test if weight rebalance and SB noise stack additively."
with open(Path(__file__).parent / "configs" / "620_spectral_v7_combo_brownian.json", "w", encoding="utf-8") as f:
    json.dump(v7, f, indent=2, ensure_ascii=False)
print("V7 written: V2 weights + Brownian sigma=0.1")

# V8: V2 + V4 combo — w_ll=0.3, w_hh=1.5, 24ep, lr=1e-4
v8 = json.loads(json.dumps(base))
v8["bridge"]["spectral_w_ll"] = 0.3
v8["bridge"]["spectral_w_hh"] = 1.5
v8["training"]["num_epochs"] = 24
v8["training"]["learning_rate"] = 0.0001
v8["training"]["save_interval"] = 4
v8["checkpoint"]["save_dir"] = "./exp/620_spectral_v8_combo_long"
v8["ablation"]["name"] = "620_spectral_v8_combo_long"
v8["ablation"]["axis"] = "620_spectral_ode_v8_combo_long"
v8["ablation"]["notes"] = "B2 V8: V2+V4 combo. w_ll=0.3, w_hh=1.5 (V2 best weights) + 24 epochs + lr=1e-4 (V4 long training). Test if long training with V2 weights finds a better optimum than V2 epoch_0001."
with open(Path(__file__).parent / "configs" / "620_spectral_v8_combo_long.json", "w", encoding="utf-8") as f:
    json.dump(v8, f, indent=2, ensure_ascii=False)
print("V8 written: V2 weights + 24ep + lr=1e-4")

print("\nAll configs generated:")
for name in ["v5_ll01", "v6_ll05", "v7_combo_brownian", "v8_combo_long"]:
    p = Path(__file__).parent / "configs" / f"620_spectral_{name}.json"
    print(f"  {p.name} ({p.stat().st_size} bytes)")
