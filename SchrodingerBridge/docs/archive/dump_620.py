import json, os, glob
base = '/mnt/i/Github/Latent_Style/exp/620_spatial_bridge'
results = {}
for sj in glob.glob(os.path.join(base, '*/full_eval/*/summary.json')):
    parts = sj.split('/')
    exp = parts[8]
    ep = parts[10]
    try:
        with open(sj) as f:
            d = json.load(f)
        key = exp + '/' + ep
        results[key] = d.get('analysis', d)
    except Exception as e:
        results['ERR_' + exp] = str(e)
for sj in glob.glob(os.path.join(base, '*/full_eval_wfi/*/summary.json')):
    parts = sj.split('/')
    exp = parts[8]
    ep = parts[10]
    try:
        with open(sj) as f:
            d = json.load(f)
        key = exp + '/wfi_' + ep
        results[key] = d.get('analysis', d)
    except Exception as e:
        results['ERR_wfi_' + exp] = str(e)
for wb in glob.glob(os.path.join(base, '*/full_eval_wfi/*/wfi_benchmark.json')):
    parts = wb.split('/')
    exp = parts[8]
    ep = parts[10]
    try:
        with open(wb) as f:
            d = json.load(f)
        key = exp + '/wfi_bench_' + ep
        if key not in results: results[key] = {}
        results[key]['wfi_benchmark'] = d
    except: pass
outpath = '/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/all_620_summaries.json'
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print('OK: ' + str(len(results)) + ' entries written')
