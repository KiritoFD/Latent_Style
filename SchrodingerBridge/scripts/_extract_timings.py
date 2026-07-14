"""Extract wall_total and timings from summary.json files."""
import json
import sys
from pathlib import Path

files = [
    ("D5",   r"I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\full_eval\epoch_0005\summary.json"),
    ("P2A",  r"I:\Github\Latent_Style\SchrodingerBridge\exp\main_table\p2a_256\full_eval\epoch_0005\summary.json"),
    ("R5",   r"I:\Github\Latent_Style\SchrodingerBridge\exp\main_table\r5\full_eval\epoch_0005\summary.json"),
    ("D5_b16", r"I:\Github\Latent_Style\SchrodingerBridge\exp\infra_infer_bench\b16_save\full_eval\epoch_0005\summary.json"),
]

for name, path in files:
    p = Path(path)
    if not p.exists():
        print(f"{name}: FILE NOT FOUND ({path})")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n=== {name} ({path}) ===")
    print(f"  wall_total: {d.get('wall_total', 'N/A')}")
    print(f"  n_img:      {d.get('n_img', 'N/A')}")
    timings = d.get('timings', {})
    if timings:
        print(f"  timings:")
        for k, v in timings.items():
            print(f"    {k}: {v}")
    # Also check for any other time-related fields
    for k in sorted(d.keys()):
        if 'time' in k.lower() or 'wall' in k.lower() or 'sec' in k.lower():
            if k not in ('wall_total', 'timings'):
                print(f"  {k}: {d[k]}")
