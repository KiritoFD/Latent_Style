"""Generate V9-V12 configs targeting CLIP > 0.74.

Strategy: aggressively unlock lowfreq (w_ll) since lowfreq carries color/brightness
which is critical for CLIP style. Current spectral ODE CLIP ceiling ~0.68 with w_ll=0.3.
Need w_ll >= 1.0 to close the gap to 0.74.

V9:  w_ll=1.0, w_hh=1.5, 8ep       (full lowfreq unlock)
V10: w_ll=2.0, w_hh=1.5, 8ep       (emphasize lowfreq learning)
V11: w_ll=1.0, w_hh=2.0, 8ep       (full lowfreq unlock + strong highfreq)
V12: w_ll=1.0, w_hh=1.5, 24ep, lr=1e-4  (full lowfreq unlock + long training)
"""
import json
from pathlib import Path

BASE = Path(__file__).parent / "configs" / "620_spectral_v2_weights.json"
with open(BASE, "r", encoding="utf-8") as f:
    base = json.load(f)

configs = [
    # (name, w_ll, w_hh, num_epochs, lr, notes)
    ("v9_ll10", 1.0, 1.5, 8, 0.0002,
     "B2 V9: w_ll=1.0 (full lowfreq unlock), w_hh=1.5. Target CLIP>0.74 by letting model learn lowfreq color/brightness style transfer."),
    ("v10_ll20", 2.0, 1.5, 8, 0.0002,
     "B2 V10: w_ll=2.0 (emphasize lowfreq learning), w_hh=1.5. Test if over-weighting lowfreq pushes CLIP beyond 0.74."),
    ("v11_ll10_hh20", 1.0, 2.0, 8, 0.0002,
     "B2 V11: w_ll=1.0 (full lowfreq unlock) + w_hh=2.0 (strong highfreq). Test if combined lowfreq+highfreq emphasis maximizes CLIP."),
    ("v12_ll10_long", 1.0, 1.5, 24, 0.0001,
     "B2 V12: w_ll=1.0 (full lowfreq unlock) + 24ep + lr=1e-4. Test if long training with unlocked lowfreq finds CLIP>0.74 optimum."),
]

for name, w_ll, w_hh, num_epochs, lr, notes in configs:
    cfg = json.loads(json.dumps(base))
    cfg["bridge"]["spectral_w_ll"] = w_ll
    cfg["bridge"]["spectral_w_lh"] = 1.0
    cfg["bridge"]["spectral_w_hl"] = 1.0
    cfg["bridge"]["spectral_w_hh"] = w_hh
    cfg["training"]["num_epochs"] = num_epochs
    cfg["training"]["learning_rate"] = lr
    cfg["training"]["save_interval"] = 4 if num_epochs > 8 else 1
    cfg["checkpoint"]["save_dir"] = f"./exp/620_spectral_{name}"
    cfg["ablation"]["name"] = f"620_spectral_{name}"
    cfg["ablation"]["axis"] = f"620_spectral_ode_{name}"
    cfg["ablation"]["notes"] = notes
    out = Path(__file__).parent / "configs" / f"620_spectral_{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"V9-12 written: {out.name} w_ll={w_ll} w_hh={w_hh} epochs={num_epochs} lr={lr}")

print("\nAll configs generated for CLIP>0.74 target.")
