import csv, os, json, glob

base_eval = r'I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag\eval_curve'

print("=== SaMAM eval_curve all steps ===")
print("step,clip_style,lpips,clip_content,n")
if os.path.isdir(base_eval):
    dirs = sorted([d for d in os.listdir(base_eval) if d.startswith('step_')])
    for d in dirs:
        mpath = os.path.join(base_eval, d, 'metrics.csv')
        if os.path.isfile(mpath):
            with open(mpath, newline='') as f:
                rows = list(csv.DictReader(f))
            cs_vals = [float(x['clip_style']) for x in rows if x.get('clip_style')]
            lp_vals = [float(x['lpips']) for x in rows if x.get('lpips')]
            cc_vals = [float(x['clip_content']) for x in rows if x.get('clip_content')]
            cs_avg = sum(cs_vals)/len(cs_vals) if cs_vals else 0
            lp_avg = sum(lp_vals)/len(lp_vals) if lp_vals else 0
            cc_avg = sum(cc_vals)/len(cc_vals) if cc_vals else 0
            print(f"{d},{cs_avg:.6f},{lp_avg:.6f},{cc_avg:.6f},{len(rows)}")
        else:
            print(f"{d},NO_METRICS_CSV")

        spath = os.path.join(base_eval, d, 'summary.json')
        if os.path.isfile(spath):
            try:
                with open(spath) as f:
                    sj = json.load(f)
                overview = sj.get('analysis', {}).get('all_pairs_overview', {})
                print(f"  summary.json: clip_style={overview.get('clip_style','N/A')}, lpips={overview.get('content_lpips','N/A')}, clip_content={overview.get('clip_content','N/A')}")
            except Exception as e:
                print(f"  summary.json: ERROR reading: {e}")
else:
    print(f"Directory not found: {base_eval}")

print()
print("=== SaMAM step_002250 detail ===")
detail_path = os.path.join(base_eval, 'step_002250', 'metrics.csv')
if os.path.isfile(detail_path):
    with open(detail_path, newline='') as f:
        rows = list(csv.DictReader(f))
    cs_vals = [float(x['clip_style']) for x in rows if x.get('clip_style')]
    lp_vals = [float(x['lpips']) for x in rows if x.get('lpips')]
    cc_vals = [float(x['clip_content']) for x in rows if x.get('clip_content')]
    print(f"n={len(rows)}")
    print(f"clip_style avg={sum(cs_vals)/len(cs_vals):.6f}")
    print(f"lpips avg={sum(lp_vals)/len(lp_vals):.6f}")
    print(f"clip_content avg={sum(cc_vals)/len(cc_vals):.6f}")
    styles = set(x.get('tgt_style', '') for x in rows)
    print(f"styles={styles}")

print()
print("=== SaMAM convergence runs ===")
base_results = r'I:\Github\Latent_Style\Related_Works\baseline_pipeline\results'
if os.path.isdir(base_results):
    samam_dirs = sorted([d for d in os.listdir(base_results) if 'samam' in d.lower() and 'distinct5' in d.lower()])
    for d in samam_dirs:
        print(f"  {d}")
        full = os.path.join(base_results, d)
        for sj_path in glob.glob(os.path.join(full, '**', 'summary.json'), recursive=True):
            rel = os.path.relpath(sj_path, full)
            try:
                with open(sj_path) as f:
                    sj = json.load(f)
                overview = sj.get('analysis', {}).get('all_pairs_overview', {})
                keys = ['clip_style', 'content_lpips', 'clip_content', 'clip_t', 'fid', 'art_fid']
                filtered = {k: overview.get(k) for k in keys if overview.get(k) is not None}
                print(f"    {rel}: {json.dumps(filtered)}")
            except Exception as e:
                print(f"    {rel}: ERROR reading: {e}")

print()
print("=== SaMST convergence runs ===")
if os.path.isdir(base_results):
    samst_dirs = sorted([d for d in os.listdir(base_results) if 'samst' in d.lower() and 'distinct5' in d.lower()])
    for d in samst_dirs:
        print(f"  {d}")
        full = os.path.join(base_results, d)
        for sj_path in glob.glob(os.path.join(full, '**', 'summary.json'), recursive=True):
            rel = os.path.relpath(sj_path, full)
            try:
                with open(sj_path) as f:
                    sj = json.load(f)
                overview = sj.get('analysis', {}).get('all_pairs_overview', {})
                keys = ['clip_style', 'content_lpips', 'clip_content', 'clip_t', 'fid', 'art_fid']
                filtered = {k: overview.get(k) for k in keys if overview.get(k) is not None}
                print(f"    {rel}: {json.dumps(filtered)}")
            except Exception as e:
                print(f"    {rel}: ERROR reading: {e}")

print()
print("=== experiment_database_all.csv SaMAM/SaMST/IDT rows ===")
db_path = r'I:\experiment_database_all.csv'
if os.path.isfile(db_path):
    with open(db_path, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if any(kw in r.get('experiment', '').lower() for kw in ['samam', 'samst', 'idt'])]
    for r in rows[:60]:
        exp = r.get('experiment', '')[:60]
        ep = r.get('epoch', '?')
        cs = r.get('clip_style', '?')
        lp = r.get('content_lpips', '?')
        ds = r.get('dataset', '?')
        cc = r.get('clip_content', '?')
        print(f"  {exp:60s} e{ep:>5s} cs={cs:>8s} lp={lp:>8s} cc={cc:>8s} ds={ds}")
