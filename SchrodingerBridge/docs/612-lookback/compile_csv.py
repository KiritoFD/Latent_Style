import csv, os

base = r'G:\GitHub\Latent_Style\SchrodingerBridge'

HEADER = ['experiment_id','dataset','method','family','sub_family','batch','epoch',
    'transfer_clip_style','transfer_lpips','allpairs_clip_style','allpairs_lpips',
    'id_clip_style','id_lpips','delta_idt_transfer','delta_idt_full',
    'epoch_time_s','train_wall_s','train_wall_min','infer_ms_per_img','eval_wall_s',
    'cuda_gb','location','status','solver','tokenizer','prediction','kinetic','proximal','attention','notes']
rows = []

def read_csv(path):
    for enc in ['utf-8-sig','utf-8','latin-1']:
        try:
            with open(path,'r',newline='',encoding=enc) as f:
                return list(csv.DictReader(f))
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(path,'r',newline='',encoding='utf-8',errors='replace') as f:
        return list(csv.DictReader(f))

def mk(**kw):
    r = {k:'' for k in HEADER}
    for k,v in kw.items():
        if k in r:
            r[k] = str(v) if v is not None else ''
    return r

def safe_float(s):
    try: return float(s)
    except: return 0.0

def sf(s):
    try:
        v = float(s)
        if v == int(v): return str(int(v))
        return str(round(v,6))
    except:
        return str(s).strip()

# ===== 1. epoch-eval-table.csv =====
eet = read_csv(os.path.join(base,'docs/experiments/2026-06-07-inmortal-epoch-eval-table.csv'))
for row in eet:
    run_name = row.get('run_name','').strip()
    epoch_label = row.get('epoch','').strip()
    family = row.get('family','').strip()
    train_batch = row.get('train_batch','').strip()
    
    prediction = 'endpoint' if ('XPred' in family or 'xpred' in run_name.lower()) else 'velocity'
    
    kinetic = 'off'
    family_lower = family.lower()
    if 'manifold' in family_lower:
        if 'stokes' in family_lower: kinetic = 'manifold_adaptive+stokes'
        elif 'aniso' in family_lower: kinetic = 'manifold_adaptive+anisotropic'
        elif 'queue' in family_lower: kinetic = 'manifold_adaptive'
        else: kinetic = 'manifold_adaptive'
    elif 'spatial' in family_lower: kinetic = 'spatial_lap'
    elif 'spectral' in family_lower: kinetic = 'spectral_orthogonal'
    elif 'phighpass' in family_lower: kinetic = 'off'
    elif 'bary' in family_lower: kinetic = 'off'
    else: kinetic = 'global_l2'
    
    proximal = 'off'
    if 'phighpass' in family_lower: proximal = 'highpass_residual'
    elif 'pmod' in family_lower: proximal = 'phase_modulation'
    elif 'pattn' in family_lower and ('anisostokes' not in family_lower and 'stokes' not in family_lower): proximal = 'crossattn_texture'
    elif 'pattn' in family_lower and ('anisostokes' in family_lower or 'stokes' in family_lower): proximal = 'crossattn_texture'
    elif 'pattn' in family_lower: proximal = 'crossattn_texture'
    elif 'structot' in family_lower: proximal = 'struct_ot'
    
    eid = run_name + '/' + epoch_label
    rows.append(mk(
        experiment_id=eid, dataset='distinct5-512/full', method='LBM',
        family=family, sub_family=family.lower().replace('_',' '),
        batch=train_batch, epoch=epoch_label.replace('epoch_',''),
        transfer_clip_style=sf(row.get('clip_style','')),
        transfer_lpips=sf(row.get('content_lpips','')),
        allpairs_clip_style=sf(row.get('full_clip_style','')),
        allpairs_lpips=sf(row.get('full_content_lpips','')),
        epoch_time_s=sf(row.get('epoch_time_sec','')),
        train_wall_s=sf(row.get('train_time_sec','')),
        train_wall_min=sf(round(float(row['train_time_sec'])/60,1)) if row.get('train_time_sec','').strip() else '',
        cuda_gb=sf(row.get('cuda_peak_allocated_gb','')),
        location='Remote_3060', status='completed',
        solver='euler_legacy', tokenizer='legacy_factorized',
        prediction=prediction, kinetic=kinetic, proximal=proximal,
        attention='SemanticCrossAttn'))
print(f'1. epoch-eval: {len([r for r in rows if r["experiment_id"].startswith("aaai2027_inmortal")])} immortal rows')

# ===== 2. aaai2027_results_master.csv =====
arm = read_csv(os.path.join(base,'docs/experiments/aaai2027_results_master.csv'))
for row in arm:
    exp_id = row.get('experiment','').strip()
    metric_surface = row.get('metric_surface','').strip()
    cs = row.get('clip_style','').strip()
    cl = row.get('content_lpips','').strip()
    fcs = row.get('full_clip_style','').strip()
    fcl = row.get('full_content_lpips','').strip()
    didt = row.get('delta_idt_transfer','').strip()
    didf = row.get('delta_idt_full','').strip()
    tw = row.get('train_wall','').strip()
    ds_raw = row.get('dataset','').strip()
    method = row.get('method','').strip()
    variant = row.get('variant','').strip()
    sel = row.get('selection','').strip()
    batch = row.get('train_batch','').strip()
    epochs = row.get('train_epochs','').strip()
    decision = row.get('decision','').strip()
    notes = row.get('status','').strip()
    
    if not exp_id: continue
    
    # Skip duplicative rows already well-covered by epoch-eval
    if exp_id.startswith('inmortal_') and metric_surface=='transfer' and not variant:
        continue
    
    if 'legacy256' in ds_raw: ds = 'legacy256/50x50'
    elif 'wikiart_stress1' in ds_raw: ds = 'wikiart-stress1/1000-per-style'
    elif 'wikiart_stress2' in ds_raw: ds = 'wikiart-stress2/1000-per-style'
    elif 'wikiart512' in ds_raw: ds = 'wikiart512/5style'
    elif any(x in exp_id for x in ['_e1','_e2','_e5','step_','batch','e15','e5','b50','b300']) and 'immortal' not in exp_id and 'inmortal' not in exp_id:
        ds = 'distinct5-512/1000-per-style'
    else: ds = 'distinct5-512/full'
    
    fam = variant if variant else sel
    
    if metric_surface == 'transfer':
        rows.append(mk(
            experiment_id=exp_id, dataset=ds, method=method,
            family=fam, sub_family=fam.lower(),
            batch=batch, epoch=epochs,
            transfer_clip_style=sf(cs), transfer_lpips=sf(cl),
            allpairs_clip_style=sf(fcs), allpairs_lpips=sf(fcl),
            delta_idt_transfer=sf(didt), delta_idt_full=sf(didf),
            train_wall_min=sf(tw), status=decision,
            solver='euler_legacy' if method=='LBM' else ('OMF' if method=='SaMST' else 'SSM' if method=='SaMAM' else ''),
            tokenizer='legacy_factorized' if method=='LBM' else ('OMF_weight' if method=='SaMST' else ''),
            prediction='velocity' if method in ('LBM','SaMST') else 'SSM' if method=='SaMAM' else '',
            kinetic='global_l2' if method=='LBM' else 'off',
            attention='SemanticCrossAttn' if method=='LBM' else ('OMF' if method=='SaMST' else 'Mamba' if method=='SaMAM' else ''),
            notes=notes))
    elif metric_surface == 'reported':
        rows.append(mk(
            experiment_id=exp_id, dataset=ds, method=method,
            family=fam, sub_family=fam.lower(),
            batch=batch, epoch=epochs,
            allpairs_clip_style=sf(cs), allpairs_lpips=sf(cl),
            delta_idt_transfer=sf(didt), delta_idt_full=sf(didf),
            train_wall_min=sf(tw), status=decision,
            solver='euler_legacy' if method=='LBM' else ('OMF' if method=='SaMST' else 'SSM' if method=='SaMAM' else ''),
            tokenizer='legacy_factorized' if method=='LBM' else ('OMF_weight' if method=='SaMST' else ''),
            prediction='velocity' if method in ('LBM','SaMST') else '',
            attention='SemanticCrossAttn' if method=='LBM' else ('OMF' if method=='SaMST' else 'Mamba' if method=='SaMAM' else ''),
            notes=notes))
print(f'2. results_master: {len(rows)} total rows')

# ===== 3. aaai2027_master_experiment_log.csv =====
mel = read_csv(os.path.join(base,'docs/experiments/aaai2027_master_experiment_log.csv'))
for row in mel:
    ds_raw = row.get('dataset','').strip()
    method = row.get('method','').strip()
    variant = row.get('variant_or_point','').strip()
    exp_id = f"{ds_raw}__{method}__{variant}" if ds_raw and method else ''
    cs = row.get('clip_style','').strip()
    cl = row.get('content_lpips','').strip()
    didt = row.get('delta_idt_transfer','').strip()
    didf = row.get('delta_idt_full','').strip()
    tw = row.get('train_wall','').strip()
    decision = row.get('keep_decision','').strip()
    notes = row.get('note','').strip()
    family = row.get('family','').strip()
    
    if not exp_id.strip('_'): continue
    
    if 'distinct5_512' in ds_raw: ds = 'distinct5-512/full'
    elif 'legacy256' in ds_raw: ds = 'legacy256/50x50'
    else: ds = ds_raw
    
    rows.append(mk(
        experiment_id=exp_id, dataset=ds, method=method,
        family=family, sub_family=variant,
        allpairs_clip_style=sf(cs), allpairs_lpips=sf(cl),
        delta_idt_transfer=sf(didt), delta_idt_full=sf(didf),
        train_wall_min=sf(tw.split('s')[0]) if 's' in str(tw) else '',
        status=decision,
        solver='euler_legacy' if method in ('LBM','LBM_vs_SaMAM_vs_SaMST') else ('OMF' if method=='SaMST' else ''),
        tokenizer='legacy_factorized' if method in ('LBM','LBM_vs_SaMAM_vs_SaMST') else '',
        prediction='velocity',
        notes=notes[:250] if notes else ''))
print(f'3. experiment_log: {len(rows)} total rows')

# ===== 4. aaai2027_inmortal_results_master.csv =====
irm = read_csv(os.path.join(base,'docs/experiments/aaai2027_inmortal_results_master.csv'))
for row in irm:
    exp_id = row.get('experiment','').strip()
    if not exp_id: continue
    tcs = row.get('transfer_clip_style','').strip()
    tcl = row.get('transfer_content_lpips','').strip()
    fcs = row.get('full_clip_style','').strip()
    fcl = row.get('full_content_lpips','').strip()
    family = row.get('family','').strip()
    batch = row.get('train_batch','').strip()
    epochs = row.get('train_epochs','').strip()
    selection = row.get('selection','').strip()
    reading = row.get('reading','').strip()
    
    prediction = 'endpoint' if 'xpred' in exp_id.lower() else 'velocity'
    
    rows.append(mk(
        experiment_id=exp_id + '/selected',
        dataset='distinct5-512/full', method='LBM',
        family=family, sub_family=family.lower().replace('_',' '),
        batch=batch, epoch=selection.replace('epoch_',''),
        transfer_clip_style=sf(tcs), transfer_lpips=sf(tcl),
        allpairs_clip_style=sf(fcs), allpairs_lpips=sf(fcl),
        status='completed', prediction=prediction,
        notes=reading[:200] if reading else ''))
print(f'4. inmortal_results_master: {len(rows)} total rows')
print('Done with Phase 1. Rows count will be counted at end.')
