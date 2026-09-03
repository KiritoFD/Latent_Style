import json, os

methods = ['ours_epoch_0007', 'samst_strict', 's2wat_strict']
data_dir = r'G:\GitHub\Latent_Style\Related_Works\run_511\complete_750'
# Also add styleid and adain if protocol file exists

print(f"{'Method':<20} {'CLIP-S':>8} {'LPIPS':>8} {'EC':>8} {'CLIP-S':>8} {'LPIPS':>8} {'EC':>8}")
print(f"{'':20} {'---photo->art---':>24} {'---non-identity---':>24}")
print("-"*70)

for method in methods:
    path = os.path.join(data_dir, method, 'eval_protocol750_sbmatch.json')
    if not os.path.exists(path):
        print(f"{method:<20} {'N/A':>8}")
        continue
    d = json.load(open(path))
    
    # photo->art
    pa = {'clip_style': [], 'content_lpips': []}
    for src, tgts in d.items():
        if src.lower() != 'photo': continue
        for tgt, m in tgts.items():
            if tgt.lower() == 'photo': continue
            pa['clip_style'].append(m['clip_style'])
            pa['content_lpips'].append(m['content_lpips'])
    
    pa_avg = {k: sum(v)/len(v) for k, v in pa.items()}
    pa_ec = pa_avg['clip_style'] * (1 - pa_avg['content_lpips'])
    
    # non-identity
    ni = {'clip_style': [], 'content_lpips': []}
    for src, tgts in d.items():
        for tgt, m in tgts.items():
            if src == tgt: continue
            ni['clip_style'].append(m['clip_style'])
            ni['content_lpips'].append(m['content_lpips'])
    
    ni_avg = {k: sum(v)/len(v) for k, v in ni.items()}
    ni_ec = ni_avg['clip_style'] * (1 - ni_avg['content_lpips'])
    
    print(f"{method:<20} {pa_avg['clip_style']:8.4f} {pa_avg['content_lpips']:8.4f} {pa_ec:8.4f} {ni_avg['clip_style']:8.4f} {ni_avg['content_lpips']:8.4f} {ni_ec:8.4f}")
