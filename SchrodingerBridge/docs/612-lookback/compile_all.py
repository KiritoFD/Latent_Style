import csv, os

base = r'G:\GitHub\Latent_Style\SchrodingerBridge'

HEADER = ['experiment_id','dataset','method','family','sub_family','batch','epoch',
    'transfer_clip_style','transfer_lpips','allpairs_clip_style','allpairs_lpips',
    'id_clip_style','id_lpips','delta_idt_transfer','delta_idt_full',
    'epoch_time_s','train_wall_s','train_wall_min','infer_ms_per_img','eval_wall_s',
    'cuda_gb','location','status','solver','tokenizer','prediction','kinetic','proximal','attention','notes']

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

def sf(s):
    try:
        v = float(s)
        if v == int(v): return str(int(v))
        return str(round(v,6))
    except:
        return str(s).strip()

rows = []

# ===== 1. epoch-eval-table.csv =====
eet = read_csv(os.path.join(base,'docs/experiments/2026-06-07-inmortal-epoch-eval-table.csv'))
for row in eet:
    run_name = row.get('run_name','').strip()
    epoch_label = row.get('epoch','').strip()
    family = row.get('family','').strip()
    train_batch = row.get('train_batch','').strip()
    
    prediction = 'endpoint' if ('XPred' in family or 'xpred' in run_name.lower()) else 'velocity'
    
    kinetic = 'off'
    fl = family.lower()
    if 'manifold' in fl:
        if 'stokes' in fl and 'aniso' not in fl: kinetic = 'manifold_adaptive+stokes'
        elif 'anisostokes' in fl: kinetic = 'manifold_adaptive+aniso+stokes'
        elif 'aniso' in fl: kinetic = 'manifold_adaptive+anisotropic'
        else: kinetic = 'manifold_adaptive'
    elif 'spatial' in fl: kinetic = 'spatial_lap'
    elif 'spectral' in fl: kinetic = 'spectral_orthogonal'
    elif 'phighpass' in fl: kinetic = 'off'
    elif 'bary' in fl: kinetic = 'off'
    else: kinetic = 'global_l2'
    
    proximal = 'off'
    if 'phighpass' in fl: proximal = 'highpass_residual'
    elif 'pmod' in fl: proximal = 'phase_modulation'
    elif 'pattn' in fl: proximal = 'crossattn_texture'
    elif 'structot' in fl: proximal = 'struct_ot'
    
    eid = run_name + '/' + epoch_label
    tws = row.get('train_time_sec','').strip()
    twm = str(round(float(tws)/60,1)) if tws else ''
    rows.append(mk(
        experiment_id=eid, dataset='distinct5-512/full', method='LBM',
        family=family, sub_family=family.lower().replace('_',' '),
        batch=train_batch, epoch=epoch_label.replace('epoch_',''),
        transfer_clip_style=sf(row.get('clip_style','')),
        transfer_lpips=sf(row.get('content_lpips','')),
        allpairs_clip_style=sf(row.get('full_clip_style','')),
        allpairs_lpips=sf(row.get('full_content_lpips','')),
        epoch_time_s=sf(row.get('epoch_time_sec','')),
        train_wall_s=sf(tws), train_wall_min=sf(twm),
        cuda_gb=sf(row.get('cuda_peak_allocated_gb','')),
        location='Remote_3060', status='completed',
        solver='euler_legacy', tokenizer='legacy_factorized',
        prediction=prediction, kinetic=kinetic, proximal=proximal,
        attention='SemanticCrossAttn'))
print(f'1. epoch-eval: {len(rows)} rows')

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
    # Skip inmortal rows that are better covered by epoch-eval
    if exp_id.startswith('inmortal_') and metric_surface=='transfer' and not variant: continue
    
    if 'legacy256' in ds_raw: ds = 'legacy256/50x50'
    elif 'wikiart_stress1' in ds_raw: ds = 'wikiart-stress1/1000-per-style'
    elif 'wikiart_stress2' in ds_raw: ds = 'wikiart-stress2/1000-per-style'
    elif 'wikiart512' in ds_raw: ds = 'wikiart512/5style'
    elif 'immortal' in exp_id or 'inmortal' in exp_id: ds = 'distinct5-512/full'
    else: ds = 'distinct5-512/1000-per-style'
    
    fam = variant if variant else sel
    solv = 'euler_legacy' if method=='LBM' else ('OMF' if method=='SaMST' else 'SSM' if method=='SaMAM' else '')
    tok = 'legacy_factorized' if method=='LBM' else ('OMF_weight' if method=='SaMST' else '')
    pred = 'velocity' if method in ('LBM','SaMST') else 'SSM' if method=='SaMAM' else ''
    attn = 'SemanticCrossAttn' if method=='LBM' else ('OMF' if method=='SaMST' else 'Mamba' if method=='SaMAM' else '')
    
    if metric_surface == 'transfer':
        rows.append(mk(
            experiment_id=exp_id, dataset=ds, method=method,
            family=fam, sub_family=fam.lower(),
            batch=batch, epoch=epochs,
            transfer_clip_style=sf(cs), transfer_lpips=sf(cl),
            allpairs_clip_style=sf(fcs), allpairs_lpips=sf(fcl),
            delta_idt_transfer=sf(didt), delta_idt_full=sf(didf),
            train_wall_min=sf(tw), status=decision,
            solver=solv, tokenizer=tok, prediction=pred,
            kinetic='global_l2' if method=='LBM' else 'off',
            attention=attn, notes=notes))
    else:
        rows.append(mk(
            experiment_id=exp_id, dataset=ds, method=method,
            family=fam, sub_family=fam.lower(),
            batch=batch, epoch=epochs,
            allpairs_clip_style=sf(cs), allpairs_lpips=sf(cl),
            delta_idt_transfer=sf(didt), delta_idt_full=sf(didf),
            train_wall_min=sf(tw), status=decision,
            solver=solv, tokenizer=tok, prediction=pred,
            attention=attn, notes=notes))
print(f'2. results_master: {len(rows)} rows')

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
    cp = row.get('checkpoint_or_step','').strip()
    
    if not exp_id.strip('_'): continue
    
    if 'distinct5_512' in ds_raw: ds = 'distinct5-512/full'
    elif 'legacy256' in ds_raw: ds = 'legacy256/50x50'
    else: ds = ds_raw
    
    twm = tw.split('s')[0] if 's' in str(tw) else ''
    try: twm = str(round(float(twm)/60,1)) if twm.replace('.','').isdigit() else twm
    except: pass
    
    rows.append(mk(
        experiment_id=exp_id, dataset=ds, method=method,
        family=family, sub_family=variant, epoch=cp,
        allpairs_clip_style=sf(cs), allpairs_lpips=sf(cl),
        delta_idt_transfer=sf(didt), delta_idt_full=sf(didf),
        train_wall_min=twm, status=decision,
        solver='euler_legacy' if method in ('LBM','LBM_vs_SaMAM_vs_SaMST') else ('OMF' if method=='SaMST' else ''),
        tokenizer='legacy_factorized' if method in ('LBM','LBM_vs_SaMAM_vs_SaMST') else '',
        prediction='velocity',
        notes=notes[:250] if notes else ''))
print(f'3. experiment_log: {len(rows)} rows')

# ===== 4. inmortal_results_master.csv =====
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
    status_r = row.get('status','').strip()
    
    prediction = 'endpoint' if 'xpred' in exp_id.lower() else 'velocity'
    
    rows.append(mk(
        experiment_id=exp_id + '/selected',
        dataset='distinct5-512/full', method='LBM',
        family=family, sub_family=family.lower().replace('_',' '),
        batch=batch, epoch=selection.replace('epoch_',''),
        transfer_clip_style=sf(tcs), transfer_lpips=sf(tcl),
        allpairs_clip_style=sf(fcs), allpairs_lpips=sf(fcl),
        status=status_r, prediction=prediction,
        notes=reading[:200] if reading else ''))
print(f'4. inmortal_results: {len(rows)} rows')

# ===== 5. distinct5_same_cost_inventory.csv =====
dsci = read_csv(os.path.join(base,'docs/experiments/2026-06-04-distinct5_same_cost_inventory.csv'))
for row in dsci:
    arm_name = row.get('arm','').strip()
    method = row.get('method','').strip()
    status = row.get('status','').strip()
    twm = row.get('train_wall_min','').strip()
    ewm = row.get('eval_wall_min','').strip()
    ims = row.get('inference_ms_per_img','').strip()
    tcs = row.get('transfer_clip_style','').strip()
    tlp = row.get('transfer_lpips','').strip()
    fcs = row.get('full_clip_style','').strip()
    flp = row.get('full_lpips','').strip()
    didt = row.get('delta_idt_transfer','').strip()
    
    ds = 'distinct5-512/1000-per-style'
    exp_id = f"{ds.replace('/','_')}__{method}__{arm_name}"
    
    # Map arm name to epoch
    epoch = '1'
    if 'e2' in arm_name: epoch = '2'
    elif 'e5' in arm_name: epoch = '5'
    elif 'e15' in arm_name: epoch = '15'
    elif 'step_2250' in arm_name: epoch = '2250'
    elif 'step_3000' in arm_name: epoch = '3000'
    elif 'batch' in arm_name: epoch = arm_name.split('_')[0].replace('b','').replace('batch','') if arm_name else ''
    
    family = arm_name
    solv = 'euler_legacy' if method=='LBM' else ('OMF' if method=='SaMST' else 'SSM' if method=='SaMAM' else '')
    tok = 'legacy_factorized' if method=='LBM' else ('OMF_weight' if method=='SaMST' else '')
    pred = 'velocity' if method in ('LBM','SaMST') else 'SSM' if method=='SaMAM' else ''
    attn = 'SemanticCrossAttn' if method=='LBM' else ('OMF' if method=='SaMST' else 'Mamba' if method=='SaMAM' else '')
    
    rows.append(mk(
        experiment_id=exp_id, dataset=ds, method=method,
        family=family, sub_family=arm_name.lower(),
        batch='44' if method=='LBM' else ('6' if method=='SaMAM' else ''),
        epoch=epoch,
        transfer_clip_style=sf(tcs), transfer_lpips=sf(tlp),
        allpairs_clip_style=sf(fcs), allpairs_lpips=sf(flp),
        delta_idt_transfer=sf(didt),
        train_wall_min=sf(twm), eval_wall_s=sf(ewm),
        infer_ms_per_img=sf(ims),
        location='Remote_3060' if 'remote' in arm_name.lower() or method=='LBM' else 'Local_Win',
        status=status, solver=solv, tokenizer=tok, prediction=pred,
        kinetic='global_l2' if method=='LBM' else 'off',
        attention=attn))
print(f'5. same_cost_inventory: {len(rows)} rows')

# ===== 6. selected_style_metrics_historical_merged.csv =====
ssm = read_csv(os.path.join(base,'docs/experiments/comparison_20260602/selected_style_metrics_historical_merged.csv'))
for row in ssm:
    method = row.get('method','').strip()
    run_label = row.get('run','').strip()
    cs_up = row.get('clip_style_up','').strip()
    lpips_down = row.get('lpips_down','').strip()
    
    ds = 'legacy256/50x50'
    exp_id = f"{ds.replace('/','_')}__{method}__{run_label}"
    
    rows.append(mk(
        experiment_id=exp_id, dataset=ds, method=method,
        family=method, sub_family=run_label,
        allpairs_clip_style=sf(cs_up), allpairs_lpips=sf(lpips_down),
        status='completed',
        solver='euler_legacy' if method=='Ours' else ('OMF' if method=='SaMST' else ''),
        tokenizer='legacy_factorized' if method=='Ours' else ('OMF_weight' if method=='SaMST' else ''),
        prediction='velocity'))
print(f'6. historical_merged: {len(rows)} rows')

# ===== 7. inmortal-ceiling-summary.csv =====
ics = read_csv(os.path.join(base,'docs/experiments/2026-06-07-inmortal-ceiling-summary.csv'))
for row in ics:
    run_name = row.get('run','').strip()
    method = row.get('method','').strip()
    tcs = row.get('transfer_clip_style','').strip()
    tcl = row.get('transfer_content_lpips','').strip()
    fcs = row.get('full_clip_style','').strip()
    fcl = row.get('full_content_lpips','').strip()
    selection = row.get('selection','').strip()
    batch = row.get('train_batch','').strip()
    epochs = row.get('train_epochs','').strip()
    reading = row.get('reading','').strip()
    
    prediction = 'endpoint' if 'xpred' in run_name.lower() else 'velocity'
    
    rows.append(mk(
        experiment_id=run_name + '/ceiling',
        dataset='distinct5-512/full', method='LBM',
        family=method, sub_family=run_name,
        batch=batch, epoch=selection.replace('epoch_',''),
        transfer_clip_style=sf(tcs), transfer_lpips=sf(tcl),
        allpairs_clip_style=sf(fcs), allpairs_lpips=sf(fcl),
        status='completed', prediction=prediction,
        notes=reading[:200] if reading else ''))
print(f'7. ceiling-summary: {len(rows)} rows')

# ===== 8. inmortal-stage-summary.csv =====
iss = read_csv(os.path.join(base,'docs/experiments/2026-06-07-inmortal-stage-summary.csv'))
for row in iss:
    run_name = row.get('run_name','').strip()
    cs = row.get('clip_style','').strip()
    cl = row.get('content_lpips','').strip()
    selection = row.get('selection','').strip()
    family = row.get('family','').strip()
    batch = row.get('train_batch','').strip()
    epochs = row.get('train_epochs','').strip()
    fcs = row.get('selected_full_clip_style','').strip()
    fcl = row.get('selected_full_content_lpips','').strip()
    
    prediction = 'endpoint' if 'xpred' in run_name.lower() else 'velocity'
    
    rows.append(mk(
        experiment_id=run_name + '/stage',
        dataset='distinct5-512/full', method='LBM',
        family=family, sub_family=family.lower().replace('_',' '),
        batch=batch, epoch=selection.replace('epoch_',''),
        transfer_clip_style=sf(cs), transfer_lpips=sf(cl),
        allpairs_clip_style=sf(fcs), allpairs_lpips=sf(fcl),
        status='completed', prediction=prediction,
        notes=f'checkpoint_count={row.get("checkpoint_count","")} evaluated_count={row.get("evaluated_count","")}'))
print(f'8. stage-summary: {len(rows)} rows')

# ===== 9. distinct5_latent_baseline_convergence.csv =====
dlb = read_csv(os.path.join(base,'docs/experiments/2026-06-07-distinct5_latent_baseline_convergence.csv'))
for row in dlb:
    method = row.get('method','').strip()
    point_type = row.get('point_type','').strip()
    selection = row.get('selection','').strip()
    tcs = row.get('transfer_clip_style','').strip()
    tcl = row.get('transfer_content_lpips','').strip()
    fcs = row.get('full_clip_style','').strip()
    fcl = row.get('full_content_lpips','').strip()
    didt = row.get('delta_idt_transfer','').strip()
    reading = row.get('reading','').strip()
    
    exp_id = f"distinct5_512__{method}__{selection}"
    m = method.split('-')[0] if '-' in method else method
    
    rows.append(mk(
        experiment_id=exp_id,
        dataset='distinct5-512/1000-per-style', method=m,
        family=method, sub_family=selection,
        transfer_clip_style=sf(tcs), transfer_lpips=sf(tcl),
        allpairs_clip_style=sf(fcs), allpairs_lpips=sf(fcl),
        delta_idt_transfer=sf(didt),
        status='keep', 
        solver='SSM' if 'MAM' in m else 'OMF',
        tokenizer='SSM' if 'MAM' in m else 'OMF_weight',
        prediction='velocity',
        notes=reading[:200] if reading else ''))
print(f'9. latent_baseline: {len(rows)} rows')

# ===== 10. noop_comparison keypoints (3 files) =====
for fname, ds_map in [
    ('distinct5_512_keypoints.csv', 'distinct5-512/1000-per-style'),
    ('wikiart512_5style_keypoints.csv', 'wikiart512/5style'),
    ('legacy256_overfit50_keypoints.csv', 'legacy256/50x50')]:
    nkp = read_csv(os.path.join(base, 'docs/experiments/noop_comparison_across_datasets_20260602', fname))
    for row in nkp:
        method = row.get('method','').strip()
        label = row.get('label','').strip()
        scope = row.get('scope','').strip()
        cs = row.get('clip_style','').strip()
        cl = row.get('content_lpips','').strip()
        step = row.get('step_or_epoch','').strip()
        sg = row.get('style_gain_vs_noop','').strip()
        
        exp_id = f"{ds_map.replace('/','_')}__{method}__{label}__{scope}"
        
        attn = 'SemanticCrossAttn' if 'LANCET' in method else ('Mamba' if 'SaMAM' in method else 'OMF' if 'SaMST' in method else '')
        solv = 'euler_legacy' if 'LANCET' in method else ('SSM' if 'SaMAM' in method else 'OMF' if 'SaMST' in method else '')
        tok = 'legacy_factorized' if 'LANCET' in method else ('SSM' if 'SaMAM' in method else 'OMF_weight' if 'SaMST' in method else '')
        pred = 'velocity' if method in ('LANCET','SaMST') else ('SSM' if method=='SaMAM' else '')
        
        metric_field = 'allpairs' if scope=='full' else 'transfer'
        
        if metric_field == 'transfer':
            rows.append(mk(
                experiment_id=exp_id, dataset=ds_map,
                method='LBM' if 'LANCET' in method else method,
                family=method, sub_family=label,
                epoch=step,
                transfer_clip_style=sf(cs), transfer_lpips=sf(cl),
                delta_idt_transfer=sf(sg),
                status='keep', solver=solv, tokenizer=tok,
                prediction=pred, attention=attn))
        else:
            rows.append(mk(
                experiment_id=exp_id, dataset=ds_map,
                method='LBM' if 'LANCET' in method else method,
                family=method, sub_family=label,
                epoch=step,
                allpairs_clip_style=sf(cs), allpairs_lpips=sf(cl),
                delta_idt_full=sf(sg),
                status='keep', solver=solv, tokenizer=tok,
                prediction=pred, attention=attn))
print(f'10. noop keypoints: {len(rows)} rows')

# ===== 11. scatter_points.csv =====
sp = read_csv(os.path.join(base,'docs/experiments/comparison_20260602/scatter_points.csv'))
for row in sp:
    ds_label = row.get('dataset','').strip()
    scope = row.get('scope','').strip()
    method = row.get('method','').strip()
    label = row.get('label','').strip()
    step = row.get('step','').strip()
    cs = row.get('clip_style','').strip()
    cl = row.get('content_lpips','').strip()
    point_kind = row.get('point_kind','').strip()
    
    if 'legacy256' in ds_label: ds = 'legacy256/50x50'
    elif 'wikiart512' in ds_label: ds = 'wikiart512/5style'
    elif 'distinct5' in ds_label: ds = 'distinct5-512/1000-per-style'
    else: ds = ds_label
    
    exp_id = f"{ds.replace('/','_')}__{method}__{label}__{scope}"
    
    metric_field = 'allpairs' if scope=='full' else 'transfer'
    
    attn = 'SemanticCrossAttn' if 'LANCET' in method else ('Mamba' if 'SaMAM' in method else 'OMF' if 'SaMST' in method else '')
    solv = 'euler_legacy' if 'LANCET' in method else ('SSM' if 'SaMAM' in method else 'OMF' if 'SaMST' in method else '')
    tok = 'legacy_factorized' if 'LANCET' in method else ('SSM' if 'SaMAM' in method else 'OMF_weight' if 'SaMST' in method else '')
    pred = 'velocity' if method in ('LANCET','SaMST') else ('SSM' if method=='SaMAM' else '')
    
    if metric_field == 'transfer':
        rows.append(mk(
            experiment_id=exp_id, dataset=ds, method='LBM' if 'LANCET' in method else method,
            family=method, sub_family=label,
            epoch=step,
            transfer_clip_style=sf(cs), transfer_lpips=sf(cl),
            status='keep', solver=solv, tokenizer=tok,
            prediction=pred, attention=attn,
            notes=point_kind))
    else:
        rows.append(mk(
            experiment_id=exp_id, dataset=ds, method='LBM' if 'LANCET' in method else method,
            family=method, sub_family=label,
            epoch=step,
            allpairs_clip_style=sf(cs), allpairs_lpips=sf(cl),
            status='keep', solver=solv, tokenizer=tok,
            prediction=pred, attention=attn,
            notes=point_kind))
print(f'11. scatter_points: {len(rows)} rows')

# ===== 12. lancet_representative_points.csv =====
lrp = read_csv(os.path.join(base,'docs/experiments/comparison_20260602/lancet_representative_points.csv'))
for row in lrp:
    ds_label = row.get('dataset','').strip()
    method = row.get('method','').strip()
    label = row.get('label','').strip()
    stage = row.get('stage','').strip()
    scope = row.get('scope','').strip()
    cs = row.get('clip_style','').strip()
    cl = row.get('content_lpips','').strip()
    
    if 'legacy256' in ds_label: ds = 'legacy256/50x50'
    elif 'wikiart512' in ds_label: ds = 'wikiart512/5style'
    elif 'distinct5' in ds_label: ds = 'distinct5-512/1000-per-style'
    else: ds = ds_label
    
    exp_id = f"{ds.replace('/','_')}__{method}__{label}__{scope}__rep"
    
    metric_field = 'allpairs' if scope=='full' else 'transfer'
    
    if metric_field == 'transfer':
        rows.append(mk(
            experiment_id=exp_id, dataset=ds, method=method,
            family=method, sub_family=label,
            transfer_clip_style=sf(cs), transfer_lpips=sf(cl),
            status='keep', solver='euler_legacy', tokenizer='legacy_factorized',
            prediction='velocity', attention='SemanticCrossAttn',
            notes=stage))
    else:
        rows.append(mk(
            experiment_id=exp_id, dataset=ds, method=method,
            family=method, sub_family=label,
            allpairs_clip_style=sf(cs), allpairs_lpips=sf(cl),
            status='keep', solver='euler_legacy', tokenizer='legacy_factorized',
            prediction='velocity', attention='SemanticCrossAttn',
            notes=stage))
print(f'12. lancet_rep: {len(rows)} rows')

# ===== 13. clip_style_vs_1lpips_full_transfer_points.csv =====
cpt = read_csv(os.path.join(base,'docs/experiments/distinct5_512_20260602/tables/clip_style_vs_1lpips_full_transfer_points.csv'))
for row in cpt:
    scope = row.get('scope','').strip()
    family = row.get('family','').strip()
    label = row.get('label','').strip()
    step = row.get('step_or_epoch','').strip()
    cs = row.get('clip_style','').strip()
    cl = row.get('content_lpips','').strip()
    train_min = row.get('train_min','').strip()
    note = row.get('note','').strip()
    
    ds = 'distinct5-512/1000-per-style'
    exp_id = f"{ds.replace('/','_')}__{family}__{label}__{scope}"
    
    metric_field = 'allpairs' if scope=='full' else 'transfer'
    
    attn = 'SemanticCrossAttn' if family=='LANCET' else ('Mamba' if family=='SaMAM' else 'OMF' if family=='SaMST' else '')
    solv = 'euler_legacy' if family=='LANCET' else ('SSM' if family=='SaMAM' else 'OMF' if family=='SaMST' else '')
    tok = 'legacy_factorized' if family=='LANCET' else ('SSM' if family=='SaMAM' else 'OMF_weight' if family=='SaMST' else '')
    pred = 'velocity' if family in ('LANCET','SaMST') else ('SSM' if family=='SaMAM' else '')
    
    if metric_field == 'transfer':
        rows.append(mk(
            experiment_id=exp_id, dataset=ds, method='LBM' if family=='LANCET' else family,
            family=family, sub_family=label,
            epoch=step,
            transfer_clip_style=sf(cs), transfer_lpips=sf(cl),
            train_wall_min=sf(train_min),
            status='keep', solver=solv, tokenizer=tok,
            prediction=pred, attention=attn,
            notes=note))
    else:
        rows.append(mk(
            experiment_id=exp_id, dataset=ds, method='LBM' if family=='LANCET' else family,
            family=family, sub_family=label,
            epoch=step,
            allpairs_clip_style=sf(cs), allpairs_lpips=sf(cl),
            train_wall_min=sf(train_min),
            status='keep', solver=solv, tokenizer=tok,
            prediction=pred, attention=attn,
            notes=note))
print(f'13. clip_style_points: {len(rows)} rows')

# ===== 14. master_experiment_inventory.csv (phase2 rows) =====
mei = read_csv(os.path.join(base,'docs/612-lookback/master_experiment_inventory.csv'))
for row in mei:
    sub_family = row.get('sub_family','').strip()
    # Take all phase2 rows + also add rows that are already there
    if not sub_family:
        continue
    exp_id = row.get('dataset','').strip() + '__' + row.get('method','').strip() + '__' + row.get('family','').strip() + '__' + row.get('sub_family','').strip()
    rows.append(mk(
        experiment_id=exp_id,
        dataset=row.get('dataset','').strip(),
        method=row.get('method','').strip(),
        family=row.get('family','').strip(),
        sub_family=row.get('sub_family','').strip(),
        batch=row.get('batch','').strip(),
        epoch=row.get('epoch','').strip(),
        transfer_clip_style=sf(row.get('transfer_clip_style','')),
        transfer_lpips=sf(row.get('transfer_lpips','')),
        allpairs_clip_style=sf(row.get('allpairs_clip_style','')),
        allpairs_lpips=sf(row.get('allpairs_lpips','')),
        id_clip_style=sf(row.get('id_clip_style','')),
        id_lpips=sf(row.get('id_lpips','')),
        delta_idt_transfer=sf(row.get('delta_idt_transfer','')),
        delta_idt_full=sf(row.get('delta_idt_full','')),
        epoch_time_s=sf(row.get('epoch_time_s','')),
        train_wall_s=sf(row.get('train_wall_s','')),
        train_wall_min=sf(row.get('train_wall_min','')),
        infer_ms_per_img=sf(row.get('infer_ms_per_img','')),
        eval_wall_s=sf(row.get('eval_wall_s','')),
        location=row.get('location','').strip(),
        status=row.get('status','').strip(),
        solver=row.get('solver','').strip(),
        tokenizer=row.get('tokenizer','').strip(),
        prediction=row.get('prediction','').strip(),
        kinetic=row.get('kinetic','').strip(),
        proximal=row.get('proximal','').strip(),
        attention=row.get('attention','').strip(),
        notes=row.get('notes','').strip()))
print(f'14. master_inventory: {len(rows)} rows')

# ===== 15. timing data =====
tim = read_csv(os.path.join(base,'docs/timing/distinct5_same_cost_20260605.csv'))
for row in tim:
    method = row.get('method','').strip()
    label = row.get('label','').strip()
    tws = row.get('train_wall_seconds','').strip()
    twm = row.get('train_minutes','').strip()
    iws = row.get('infer_wall_seconds','').strip()
    ims = row.get('infer_ms_per_image','').strip()
    tcs = row.get('transfer_clip_style','').strip()
    tcl = row.get('transfer_content_lpips','').strip()
    didt = row.get('transfer_delta_idt','').strip()
    
    exp_id = f"distinct5_512_timing__{method}__{label}"
    
    attn = 'SemanticCrossAttn' if method=='LBM' else ('Mamba' if method=='SaMAM' else 'OMF' if method=='SaMST' else '')
    solv = 'euler_legacy' if method=='LBM' else ('SSM' if method=='SaMAM' else 'OMF' if method=='SaMST' else '')
    tok = 'legacy_factorized' if method=='LBM' else ('SSM' if method=='SaMAM' else 'OMF_weight' if method=='SaMST' else '')
    
    rows.append(mk(
        experiment_id=exp_id,
        dataset='distinct5-512/1000-per-style', method=method,
        family=method, sub_family=label,
        transfer_clip_style=sf(tcs), transfer_lpips=sf(tcl),
        delta_idt_transfer=sf(didt),
        train_wall_s=sf(tws), train_wall_min=sf(twm),
        infer_ms_per_img=sf(ims),
        status='keep', solver=solv, tokenizer=tok,
        prediction='velocity', attention=attn))
print(f'15. timing: {len(rows)} rows')

# ===== Final Output =====
# Deduplicate by experiment_id (keep first)
seen = set()
final = []
for r in rows:
    eid = r['experiment_id']
    if eid not in seen:
        seen.add(eid)
        final.append(r)

# Map dataset names to canonical form
def canon_ds(ds):
    ds = ds.strip()
    if 'distinct5-512' in ds and '1000-per' in ds: return 'distinct5-512/1000-per-style'
    if 'distinct5-512' in ds and 'full' in ds: return 'distinct5-512/full'
    if 'distinct5-512' in ds: return 'distinct5-512/1000-per-style'
    if 'wikiart-stress1' in ds or 'wikiart_stress1' in ds: return 'wikiart-stress1/1000-per-style'
    if 'wikiart-stress2' in ds or 'wikiart_stress2' in ds: return 'wikiart-stress2/1000-per-style'
    if 'wikiart512' in ds: return 'wikiart512/5style'
    if 'legacy256' in ds: return 'legacy256/50x50'
    return ds

for r in final:
    r['dataset'] = canon_ds(r['dataset'])

outpath = os.path.join(base,'docs/612-lookback/all_experiments.csv')
with open(outpath,'w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=HEADER)
    w.writeheader()
    w.writerows(final)

print(f'\n=== FINAL OUTPUT: {len(final)} rows written to {outpath} ===')
