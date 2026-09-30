"""Deep statistical analysis of EXPERIMENT_ARCHAEOLOGY_MASTER.csv"""
import csv
import os
from collections import defaultdict
import math

CSV_PATH = r"G:\GitHub\Latent_Style\EXPERIMENT_ARCHAEOLOGY_MASTER.csv"
OUT_DIR = r"G:\GitHub\Latent_Style\docs\621"

def safe_float(s):
    try:
        v = float(s)
        return v if not math.isnan(v) else None
    except:
        return None

def main():
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total rows: {len(rows)}")
    print(f"Columns: {list(rows[0].keys())}")

    # 1. Group by method
    by_method = defaultdict(list)
    for r in rows:
        m = r.get('method', '').strip() or 'unknown'
        by_method[m].append(r)

    print("\n## Per-Method Statistics\n")
    print(f"{'Method':<30} {'Count':>6} {'clip_style':>12} {'lpips':>10} {'ssim':>8}")
    print("-" * 70)
    method_stats = []
    for m, rs in sorted(by_method.items(), key=lambda x: -len(x[1])):
        cs = [safe_float(r.get('clip_style')) for r in rs]
        lp = [safe_float(r.get('content_lpips')) for r in rs]
        ss = [safe_float(r.get('ssim_y')) for r in rs]
        cs_v = [v for v in cs if v is not None]
        lp_v = [v for v in lp if v is not None]
        ss_v = [v for v in ss if v is not None]
        cs_mean = sum(cs_v)/len(cs_v) if cs_v else 0
        cs_max = max(cs_v) if cs_v else 0
        lp_mean = sum(lp_v)/len(lp_v) if lp_v else 0
        lp_min = min(lp_v) if lp_v else 0
        ss_mean = sum(ss_v)/len(ss_v) if ss_v else 0
        method_stats.append((m, len(rs), cs_mean, cs_max, lp_mean, lp_min, ss_mean))
        print(f"{m:<30} {len(rs):>6} {cs_mean:>8.4f}/{cs_max:.4f} {lp_mean:>6.4f}/{lp_min:.4f} {ss_mean:>8.4f}")

    # 2. Group by dataset_key
    by_dataset = defaultdict(list)
    for r in rows:
        d = r.get('dataset_key', '').strip() or 'unknown'
        by_dataset[d].append(r)

    print("\n## Per-Dataset Statistics\n")
    print(f"{'Dataset':<40} {'Count':>6} {'cs_mean':>10} {'cs_max':>10} {'lp_mean':>10}")
    print("-" * 80)
    for d, rs in sorted(by_dataset.items(), key=lambda x: -len(x[1]))[:20]:
        cs = [safe_float(r.get('clip_style')) for r in rs]
        lp = [safe_float(r.get('content_lpips')) for r in rs]
        cs_v = [v for v in cs if v is not None]
        lp_v = [v for v in lp if v is not None]
        cs_mean = sum(cs_v)/len(cs_v) if cs_v else 0
        cs_max = max(cs_v) if cs_v else 0
        lp_mean = sum(lp_v)/len(lp_v) if lp_v else 0
        print(f"{d:<40} {len(rs):>6} {cs_mean:>10.4f} {cs_max:>10.4f} {lp_mean:>10.4f}")

    # 3. Top variants by best clip_style
    by_variant = defaultdict(list)
    for r in rows:
        v = r.get('variant_or_run', '').strip() or 'unknown'
        by_variant[v].append(r)

    print("\n## Top 30 Variants by Best clip_style\n")
    print(f"{'Variant':<70} {'Count':>5} {'best_cs':>10} {'best_lp':>10}")
    print("-" * 100)
    variant_best = []
    for v, rs in by_variant.items():
        cs = [safe_float(r.get('clip_style')) for r in rs]
        lp = [safe_float(r.get('content_lpips')) for r in rs]
        cs_v = [x for x in cs if x is not None]
        lp_v = [x for x in lp if x is not None]
        if cs_v:
            variant_best.append((v, len(rs), max(cs_v), min(lp_v) if lp_v else None))
    variant_best.sort(key=lambda x: -x[2])
    for v, cnt, best_cs, best_lp in variant_best[:30]:
        lp_str = f"{best_lp:.4f}" if best_lp is not None else "N/A"
        print(f"{v:<70} {cnt:>5} {best_cs:>10.4f} {lp_str:>10}")

    # 4. Correlation between clip_style and content_lpips
    pairs = []
    for r in rows:
        cs = safe_float(r.get('clip_style'))
        lp = safe_float(r.get('content_lpips'))
        if cs is not None and lp is not None:
            pairs.append((cs, lp))

    if pairs:
        n = len(pairs)
        mean_cs = sum(p[0] for p in pairs) / n
        mean_lp = sum(p[1] for p in pairs) / n
        cov = sum((p[0]-mean_cs)*(p[1]-mean_lp) for p in pairs) / n
        std_cs = math.sqrt(sum((p[0]-mean_cs)**2 for p in pairs) / n)
        std_lp = math.sqrt(sum((p[1]-mean_lp)**2 for p in pairs) / n)
        corr = cov / (std_cs * std_lp) if std_cs > 0 and std_lp > 0 else 0
        print(f"\n## Correlation Analysis")
        print(f"clip_style vs content_lpips: r = {corr:.4f} (n={n})")
        print(f"clip_style: mean={mean_cs:.4f}, std={std_cs:.4f}, range=[{min(p[0] for p in pairs):.4f}, {max(p[0] for p in pairs):.4f}]")
        print(f"content_lpips: mean={mean_lp:.4f}, std={std_lp:.4f}, range=[{min(p[1] for p in pairs):.4f}, {max(p[1] for p in pairs):.4f}]")

    # 5. Historical ceiling by method
    print("\n## Historical clip_style Ceiling Over Time\n")
    print(f"{'Method':<30} {'Best CS':>10} {'Best LPIPS':>12} {'At Config'}")
    print("-" * 80)
    method_ceilings = []
    for m, rs in by_method.items():
        best_cs = 0
        best_rs = None
        for r in rs:
            cs = safe_float(r.get('clip_style'))
            if cs is not None and cs > best_cs:
                best_cs = cs
                best_rs = r
        if best_rs:
            method_ceilings.append((m, best_cs, safe_float(best_rs.get('content_lpips')), best_rs.get('variant_or_run', '')))
    method_ceilings.sort(key=lambda x: -x[1])
    for m, cs, lp, v in method_ceilings:
        lp_str = f"{lp:.4f}" if lp is not None else "N/A"
        print(f"{m:<30} {cs:>10.4f} {lp_str:>12} {v}")

    # 6. Pareto frontier
    print("\n## Pareto Frontier (clip_style↑, content_lpips↓)\n")
    pareto = []
    for cs, lp in pairs:
        dominated = False
        for cs2, lp2 in pairs:
            if cs2 >= cs and lp2 <= lp and (cs2 > cs or lp2 < lp):
                dominated = True
                break
        if not dominated:
            pareto.append((cs, lp))
    pareto.sort(key=lambda x: -x[0])
    print(f"{'clip_style':>12} {'content_lpips':>14}")
    print("-" * 28)
    for cs, lp in pareto[:20]:
        print(f"{cs:>12.4f} {lp:>14.4f}")

    print(f"\nPareto frontier size: {len(pareto)} points out of {len(pairs)} total")

    # 7. Per-source-path analysis
    by_source = defaultdict(list)
    for r in rows:
        sp = r.get('source_path', '').strip()
        if sp:
            # Extract the first directory component
            parts = sp.replace('\\', '/').split('/')
            key = parts[0] if parts else 'unknown'
            by_source[key].append(r)

    print("\n## Per-Source-Path Statistics (top 15)\n")
    for s, rs in sorted(by_source.items(), key=lambda x: -len(x[1]))[:15]:
        cs = [safe_float(r.get('clip_style')) for r in rs]
        cs_v = [v for v in cs if v is not None]
        if cs_v:
            print(f"{s:<50} n={len(rs):>5} cs_max={max(cs_v):.4f}")

if __name__ == '__main__':
    main()
