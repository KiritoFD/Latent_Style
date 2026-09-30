import json, sys
from pathlib import Path

exp = sys.argv[1] if len(sys.argv) > 1 else "5p1_shot01"
path = Path(f"G:/GitHub/Latent_Style/exp/72_fewshot/{exp}/full_eval/epoch_0010/summary.json")
s = json.loads(path.read_text(encoding="utf-8"))
mb = s["matrix_breakdown"]
styles = sorted(set(k for d in mb.values() for k in d.keys()))

print(f"=== {exp} clip_style matrix ===")
for src in styles:
    for tgt in styles:
        d = mb.get(src, {}).get(tgt, {})
        cs = d.get("clip_style", 0)
        cnt = d.get("count", 0)
        if cnt > 0:
            print(f"  {src[:15]:>15} -> {tgt[:15]:<15}: clip_s={cs:.4f}  lpips={d.get('content_lpips',0):.4f}  count={cnt}")
