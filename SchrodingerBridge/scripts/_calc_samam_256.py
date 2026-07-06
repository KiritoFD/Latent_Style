"""Extract SaMam 256 metrics from baseline_v2/eval/samam/summary.json."""
import json
from pathlib import Path

paths = [
    Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\samam\summary.json"),
    Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_reeval\samam_latent_step1000_reeval\summary.json"),
]

for p in paths:
    if not p.exists():
        print(f"[SKIP] {p} not found")
        continue
    print(f"\n=== {p.name} ===")
    data = json.loads(p.read_text(encoding="utf-8"))
    print("Top-level keys:", list(data.keys()))

    if "matrix_breakdown" in data:
        mb = data["matrix_breakdown"]
        styles = list(mb.keys())
        print(f"Styles: {styles}")

        clip_s_list = []
        lpips_list = []
        for src in styles:
            for tgt in styles:
                m = mb.get(src, {}).get(tgt)
                if m is None:
                    continue
                if src == tgt:
                    continue  # skip identity
                clip_s_list.append(m.get("clip_style", 0))
                lpips_list.append(m.get("art_fid_content_lpips", m.get("content_lpips", 0)))

        if clip_s_list:
            print(f"SaMam 256 (excluding identity, N={len(clip_s_list)}):")
            print(f"  CLIP-S = {sum(clip_s_list)/len(clip_s_list):.4f}")
            print(f"  LPIPS  = {sum(lpips_list)/len(lpips_list):.4f}")

    if "metrics_note" in data:
        # Check pool-level metrics
        for k in ["analysis", "pool", "pool_metrics"]:
            if k in data:
                pool = data[k]
                if isinstance(pool, dict):
                    print(f"\n{k} keys: {list(pool.keys())[:10]}")
                    if "style_transfer_ability" in pool:
                        sta = pool["style_transfer_ability"]
                        print(f"  CLIP-S (pool): {sta.get('clip_style')}")
                        print(f"  LPIPS  (pool): {sta.get('content_lpips', sta.get('art_fid_content_lpips'))}")
