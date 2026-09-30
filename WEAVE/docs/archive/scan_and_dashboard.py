#!/usr/bin/env python3
"""Unified experiment scanner, merger, CSV generator, and HTML dashboard builder.

Usage:
  python docs/scan_and_dashboard.py              # local-only scan
  python docs/scan_and_dashboard.py --remote      # also scan remote I: drive via SSH
  python docs/scan_and_dashboard.py --csv-only    # only regenerate CSV from existing data

Output:
  - docs/exp_unified.csv         : unified experiment CSV
  - docs/exp_dashboard_v2.html   : interactive HTML dashboard
"""
import json, os, sys, csv, subprocess, shutil
from collections import defaultdict, Counter
from datetime import datetime

# ---- CONFIG ----
LOCAL_BASE = os.path.join(os.path.dirname(__file__), '..')
LOCAL_EXP = os.path.join(LOCAL_BASE, 'exp')
LOCAL_RESULTS_CSV = os.path.join(os.path.dirname(__file__), 'exp_unified.csv')
LOCAL_DASHBOARD = os.path.join(os.path.dirname(__file__), 'exp_dashboard_v2.html')

REMOTE_HOST = "administrator@100.115.18.62"
REMOTE_PORT = "2222"
REMOTE_BASE = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
REMOTE_BASE2 = r"I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp"
REMOTE_DB = r"I:\experiment_database_all.csv"
REMOTE_SCAN_SCRIPT = r"I:\_remote_scan_v2.py"

# ---- METRIC EXTRACTION ----
def extract_from_summary(data):
    """Extract metrics from a summary.json, handling nested format."""
    m = {}
    
    # Format 1: analysis.all_pairs_overview
    if isinstance(data, dict) and 'analysis' in data:
        overview = data.get('analysis', {}).get('all_pairs_overview', {})
        if overview and isinstance(overview, dict) and isinstance(overview.get('clip_style'), (int, float)) and overview['clip_style'] > 0:
            m['clip_style'] = overview.get('clip_style')
            m['clip_s_delta_idt'] = overview.get('clip_s_delta_idt')
            m['clip_t'] = overview.get('clip_t')
            m['clip_content'] = overview.get('clip_content')
            m['content_lpips'] = overview.get('content_lpips')
            m['one_minus_lpips'] = round(1 - overview['content_lpips'], 4) if isinstance(overview.get('content_lpips'), (int, float)) else None
            m['fid'] = overview.get('fid')
            m['artfid'] = overview.get('art_fid')
            return m
        
        sta = data.get('analysis', {}).get('style_transfer_ability', {})
        if sta and isinstance(sta, dict) and isinstance(sta.get('clip_style'), (int, float)) and sta['clip_style'] > 0:
            m['clip_style'] = sta.get('clip_style')
            m['clip_s_delta_idt'] = sta.get('clip_s_delta_idt')
            m['clip_t'] = sta.get('clip_t')
            m['content_lpips'] = sta.get('content_lpips')
            m['one_minus_lpips'] = round(1 - sta['content_lpips'], 4) if isinstance(sta.get('content_lpips'), (int, float)) else None
            return m
    
    # Format 2: Direct flat format
    if isinstance(data, dict) and isinstance(data.get('clip_style'), (int, float)) and data['clip_style'] > 0:
        m['clip_style'] = data['clip_style']
        m['clip_s_delta_idt'] = data.get('clip_s_delta_idt')
        m['clip_t'] = data.get('clip_t')
        m['clip_content'] = data.get('clip_content')
        m['content_lpips'] = data.get('content_lpips')
        lpips = data.get('content_lpips')
        m['one_minus_lpips'] = round(1 - lpips, 4) if isinstance(lpips, (int, float)) else None
        m['fid'] = data.get('fid')
        return m
    
    return m

def parse_path_info(fpath, base_dir):
    """Extract experiment metadata from file path."""
    rel = os.path.relpath(fpath, base_dir)
    parts = rel.replace('\\', '/').split('/')
    
    info = {}
    info['exp_dir'] = parts[0] if len(parts) > 0 else ''
    info['sub_exp'] = '/'.join(parts[1:-1]) if len(parts) > 1 else ''
    
    for p in parts:
        if p.startswith('epoch_'):
            try:
                info['epoch'] = int(p.replace('epoch_', ''))
            except:
                pass
        if p == 'full_eval' or p.startswith('full_eval'):
            info['eval_type'] = 'full_eval'
        elif 'quick_eval' in p or 'eval_quick' in p:
            info['eval_type'] = p
    
    return info

def scan_local_dir(base_dir, exp_group):
    """Scan a local experiment directory."""
    found = []
    if not os.path.isdir(base_dir):
        return found
    
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == 'summary.json':
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                    m = extract_from_summary(data)
                    if m.get('clip_style') is not None:
                        path_info = parse_path_info(fpath, base_dir)
                        m.update(path_info)
                        m['group'] = exp_group
                        m['source'] = 'local'
                        found.append(m)
                except:
                    pass
    return found

def scan_remote():
    """Scan remote server and return results."""
    results = []
    try:
        # Upload scanner script
        local_script = os.path.join(os.path.dirname(__file__), '..', 'exp', '_remote_scan_v2.py')
        if os.path.isfile(local_script):
            subprocess.run([
                'scp', '-P', REMOTE_PORT, '-o', 'LogLevel=ERROR',
                local_script, f'{REMOTE_HOST}:{REMOTE_SCAN_SCRIPT}'
            ], timeout=30, check=True)
        
        # Run remote scanner
        proc = subprocess.run([
            'ssh', '-p', REMOTE_PORT, '-o', 'LogLevel=ERROR',
            REMOTE_HOST, f'python {REMOTE_SCAN_SCRIPT}'
        ], capture_output=True, text=True, timeout=120)
        print(proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout)
        
        # Download CSV
        remote_csv = r"I:\_all_experiments_v2.csv"
        local_csv = os.path.join(os.path.dirname(__file__), 'exp_all_results_remote.csv')
        subprocess.run([
            'scp', '-P', REMOTE_PORT, '-o', 'LogLevel=ERROR',
            f'{REMOTE_HOST}:{remote_csv}', local_csv
        ], timeout=30, check=True)
        
        # Parse remote CSV
        if os.path.isfile(local_csv):
            with open(local_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cs = row.get('clip_style', '')
                    if cs:
                        try:
                            cs_val = float(cs)
                            if cs_val > 0:
                                m = dict(row)
                                m['clip_style'] = cs_val
                                for k in ['content_lpips', 'one_minus_lpips', 'clip_s_delta_idt', 'clip_t', 'clip_content', 'fid', 'artfid']:
                                    if m.get(k):
                                        try:
                                            m[k] = float(m[k])
                                        except:
                                            m[k] = None
                                    else:
                                        m[k] = None
                                try:
                                    m['epoch'] = int(m['epoch']) if m.get('epoch') else None
                                except:
                                    m['epoch'] = None
                                results.append(m)
                        except:
                            pass
    except Exception as e:
        print(f"Remote scan failed: {e}")
    
    return results

# ---- MERGE ----
def merge_results(local_results, remote_results):
    """Merge local and remote results, deduplicating."""
    # Use a key based on group+exp_dir+epoch+eval_type for dedup
    seen = {}
    merged = []
    
    for r in local_results + remote_results:
        key = f"{r.get('group', '')}|{r.get('exp_dir', '')}|{r.get('sub_exp', '')}|{r.get('epoch', '')}|{r.get('eval_type', '')}"
        if key not in seen:
            seen[key] = r
            merged.append(r)
        else:
            # Keep the one with more data
            existing = seen[key]
            if sum(1 for v in r.values() if v is not None) > sum(1 for v in existing.values() if v is not None):
                seen[key] = r
                # Replace in merged
                merged = [x for x in merged if f"{x.get('group', '')}|{x.get('exp_dir', '')}|{x.get('sub_exp', '')}|{x.get('epoch', '')}|{x.get('eval_type', '')}" != key]
                merged.append(r)
    
    return merged

# ---- CSV OUTPUT ----
def write_csv(results, path):
    csv_cols = ['group', 'exp_dir', 'sub_exp', 'epoch', 'clip_style', 'clip_s_delta_idt', 
                'clip_t', 'content_lpips', 'one_minus_lpips', 'clip_content', 'fid', 'artfid',
                'eval_type', 'source', 'dataset']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"CSV saved to {path} ({len(results)} rows)")

# ---- HTML DASHBOARD ----
def generate_html_dashboard(results, path):
    """Generate an interactive HTML dashboard similar to phase616_live_dashboard."""
    
    # Categorize results into super-families
    FAMILY_MAP = {
        'FC-SB': ['628_ablation', '625_fc_sb', '628_infer_ablation', '628_destructive_eval', 'p4_fusion_breakout', 'fc_sb_r2', 'fig5/FC-SB'],
        'Spectral': ['620_spectral_poc', '620_spectral_v1_scale', '620_spectral_v2_weights', '620_spectral_v3_brownian', '620_spectral_v4_long', '620_spectral_v5_ll01', '620_spectral_v9_ll10', '620_spectral_v10_ll20', '620_spectral_v11_ll10_hh20', 'fig5/Spectral'],
        'LANCET': ['TC/wikiart_distinct5_ema_lancet_spectralstat_from_e16_e24_b48', 'TC/wikiart_distinct5_ema_lancet_spectralstat_from_e2_e8_b64', 'TC/wikiart_distinct5_ema_lancet_spectralstat_from_e8_e16_b48', 'DB/LANCET/LBM', 'spatial620', 'fig5/LANCET'],
        'Baseline': ['samam', 'samst', 'idt', 'manual', 'ot_rerun_auto', 'DB/SaMAM', 'DB/SaMST', 'DB/idt', 'DB/Baseline', 'fig5/SaMAM', 'fig5/SaMST', 'fig5/IDT', 'RW/protocol_eval_table', 'RW/complete_750', 'RW/eval_curve', 'RW/ablation', 'RW/unified_reeval'],
        'Manuscript': ['manuscript'],
        'RW-other': ['RW/summary_json'],
    }
    
    def get_family(group):
        for fam, groups in FAMILY_MAP.items():
            if group in groups:
                return fam
        # Heuristic fallback
        if 'fc_sb' in group or '628' in group or '625' in group or 'p4_fusion' in group:
            return 'FC-SB'
        if 'spectral' in group.lower() or '620_spectral' in group:
            return 'Spectral'
        if 'lancet' in group.lower() or 'LBM' in group:
            return 'LANCET'
        return 'Other'
    
    # Map group names to short display names
    method_map = {
        '628_ablation': 'FC-SB/628',
        '625_fc_sb': 'FC-SB/625',
        '628_infer_ablation': 'FC-SB/infer',
        '628_destructive_eval': 'FC-SB/destructive',
        'p4_fusion_breakout': 'FC-SB/p4',
        'fc_sb_r2': 'FC-SB/r2',
        '620_spectral_poc': 'Spectral/poc',
        '620_spectral_v1_scale': 'Spectral/v1',
        '620_spectral_v2_weights': 'Spectral/v2',
        '620_spectral_v3_brownian': 'Spectral/v3',
        '620_spectral_v4_long': 'Spectral/v4',
        '620_spectral_v5_ll01': 'Spectral/v5',
        '620_spectral_v9_ll10': 'Spectral/v9',
        '620_spectral_v10_ll20': 'Spectral/v10',
        '620_spectral_v11_ll10_hh20': 'Spectral/v11',
        'samam': 'SaMAM',
        'samst': 'SaMST',
        'idt': 'IDT',
        'manual': 'I2SB/manual',
        'ot_rerun_auto': 'I2SB/auto',
        'spatial620': 'LANCET/spatial620',
        'manuscript': 'Manuscript',
        'DB/LANCET/LBM': 'LANCET/LBM',
        'DB/Baseline': 'Baseline',
        'DB/SaMAM': 'SaMAM',
        'DB/SaMST': 'SaMST',
        'DB/idt': 'IDT',
        'DB/CUT': 'CUT',
    }
    # fig5 groups get their own family-based mapping
    for fam in ['LANCET', 'FC-SB', 'SaMAM', 'SaMST', 'IDT', 'Spectral', 'Other']:
        method_map[f'fig5/{fam}'] = f'fig5/{fam}'
    # Related Works groups
    method_map['RW/protocol_eval_table'] = 'RW/eval'
    method_map['RW/complete_750'] = 'RW/complete'
    method_map['RW/eval_curve'] = 'RW/eval_curve'
    method_map['RW/ablation'] = 'RW/ablation'
    method_map['RW/summary_json'] = 'RW/summary'
    method_map['RW/unified_reeval'] = 'RW/unified_reeval'
    
    # Family colors
    family_colors = {
        'FC-SB': '#F59E0B',
        'Spectral': '#5BC0EB',
        'LANCET': '#D64045',
        'Baseline': '#64748B',
        'Manuscript': '#FFD700',
        'RW-other': '#A78BFA',
        'Other': '#94a3b8',
    }
    
    # Per-method colors (for chart series)
    method_colors = {
        'FC-SB/628': '#F59E0B',
        'FC-SB/625': '#EF4444',
        'FC-SB/infer': '#EC4899',
        'FC-SB/destructive': '#A855F7',
        'FC-SB/p4': '#F97316',
        'FC-SB/r2': '#D97706',
        'Spectral/poc': '#14B8A6',
        'Spectral/v1': '#2CA58D',
        'Spectral/v2': '#5BC0EB',
        'Spectral/v3': '#00B4D8',
        'Spectral/v4': '#4DABF7',
        'Spectral/v5': '#37B24D',
        'Spectral/v9': '#84CC16',
        'Spectral/v10': '#F76707',
        'Spectral/v11': '#1098AD',
        'LANCET/LBM': '#D64045',
        'LANCET/spatial620': '#F87171',
        'SaMAM': '#2F7DB7',
        'SaMST': '#2CA02C',
        'IDT': '#8E63C0',
        'I2SB/manual': '#64748B',
        'I2SB/auto': '#94A3B8',
        'CUT': '#E48F1C',
        'Manuscript': '#FFD700',
        'fig5/LANCET': '#D6404580',
        'fig5/FC-SB': '#F59E0B80',
        'fig5/SaMAM': '#2F7DB780',
        'fig5/SaMST': '#2CA02C80',
        'fig5/IDT': '#8E63C080',
        'fig5/Spectral': '#5BC0EB80',
        'fig5/Other': '#94a3b880',
        'RW/eval': '#A78BFA',
        'RW/complete': '#8B5CF6',
        'RW/eval_curve': '#7C3AED',
        'RW/ablation': '#6D28D9',
        'RW/summary': '#C4B5FD',
        'RW/unified_reeval': '#34D399',
    }
    
    # Determine eval protocol
    def get_eval_protocol(eval_type, group, exp_dir):
        et = str(eval_type).lower()
        if 'full_eval' in et or 'full' in et:
            return 'full_eval'
        if 'quick' in et or 'smoke' in et or 'n6' in et:
            return 'quick_eval'
        if 'infer' in et or 'destructive' in et:
            return 'full_eval'  # infer/destructive ablations use full eval
        if 'formal' in et:
            return 'full_eval'
        if 'unified_reeval' in et:
            return 'full_eval'
        if 'db_record' in et:
            return 'unknown'
        # Default for known groups
        if group in ['samam', 'samst', 'idt']:
            return 'convergence'
        return 'unknown'
    
    # Determine dataset
    def get_dataset(r):
        ds = r.get('dataset', '')
        if ds:
            return ds
        group = r.get('group', '')
        if 'wikiart512' in group or 'TC/' in group or 'DB/LANCET' in group:
            return 'wikiart512'
        return 'distinct5_512'
    
    # Build data points for chart
    chart_data = []
    table_data = []
    
    for r in results:
        group = r.get('group', '')
        method = method_map.get(group, group)
        family = get_family(group)
        cs = r.get('clip_style')
        lpips = r.get('content_lpips')
        oml = r.get('one_minus_lpips')
        eval_type = r.get('eval_type', '')
        
        if cs is None or not isinstance(cs, (int, float)):
            continue
        
        if oml is None and lpips is not None and isinstance(lpips, (int, float)):
            oml = round(1 - lpips, 4)
        
        eval_proto = get_eval_protocol(eval_type, group, r.get('exp_dir', ''))
        dataset = get_dataset(r)
        
        point = {
            'method': method,
            'family': family,
            'color': method_colors.get(method, family_colors.get(family, '#94a3b8')),
            'group': group,
            'name': r.get('exp_dir', ''),
            'sub_exp': r.get('sub_exp', ''),
            'epoch': r.get('epoch'),
            'clip_style': round(cs, 4) if isinstance(cs, float) else cs,
            'clip_s_delta_idt': round(r['clip_s_delta_idt'], 4) if isinstance(r.get('clip_s_delta_idt'), (int, float)) else None,
            'clip_t': round(r['clip_t'], 4) if isinstance(r.get('clip_t'), (int, float)) else None,
            'content_lpips': round(lpips, 4) if isinstance(lpips, (int, float)) else None,
            'one_minus_lpips': round(oml, 4) if isinstance(oml, (int, float)) else None,
            'fid': r.get('fid'),
            'eval_type': eval_type,
            'eval_protocol': eval_proto,
            'source': r.get('source', ''),
            'dataset': dataset,
        }
        table_data.append(point)
        if oml is not None:
            chart_data.append(point)
    
    # Find best per method (full_eval only where available)
    best_per_method = {}
    for p in table_data:
        m = p['method']
        # Prefer full_eval over quick_eval
        if m not in best_per_method:
            best_per_method[m] = p
        else:
            existing = best_per_method[m]
            # If existing is quick and new is full, replace
            if existing.get('eval_protocol') == 'quick_eval' and p.get('eval_protocol') == 'full_eval':
                best_per_method[m] = p
            elif existing.get('eval_protocol') == p.get('eval_protocol'):
                if p['clip_style'] > existing['clip_style']:
                    best_per_method[m] = p
    
    # IDT floor reference (distinct5_512)
    idt_floor = 0.6399
    
    # Build JS data
    chart_json = json.dumps(chart_data)
    table_json = json.dumps(table_data)
    best_json = json.dumps(best_per_method)
    methods_list = json.dumps(list(method_colors.keys()))
    colors_json = json.dumps(method_colors)
    
    # Baseline references for distinct5_512
    baselines = {
        'IDT (stress2)': {'clip_style': 0.7393, 'one_minus_lpips': None, 'color': '#8E63C0'},
        'SaMAM (1000 steps)': {'clip_style': 0.7017, 'one_minus_lpips': 0.7255, 'color': '#2F7DB7'},
        'SaMST (40 steps)': {'clip_style': 0.7057, 'one_minus_lpips': 0.1625, 'color': '#2CA02C'},
        'spatial620 (LANCET)': {'clip_style': 0.7055, 'one_minus_lpips': 0.7101, 'color': '#F87171'},
    }
    baselines_json = json.dumps(baselines)
    
    # FC-SB best distinct5_512 full_eval
    fcsb_best = {
        'FC-SB/628 best (full_eval)': {'clip_style': 0.7307, 'one_minus_lpips': 0.6597, 'color': '#F59E0B'},
        'FC-SB/625 P7 (quick)': {'clip_style': 0.8289, 'one_minus_lpips': 0.7030, 'color': '#EF444480'},
    }
    fcsb_best_json = json.dumps(fcsb_best)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<title>Unified Experiment Dashboard - SchrodingerBridge</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0b0f19;
    --panel: rgba(20, 26, 41, 0.75);
    --ink: #f8fafc;
    --muted: #94a3b8;
    --line: rgba(255, 255, 255, 0.08);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    background-image: radial-gradient(circle at top center, #1e293b 0%, #0b0f19 100%);
    color: var(--ink);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    padding: 20px;
    min-height: 100vh;
  }}
  .wrap {{ width: min(1400px, calc(100vw - 32px)); margin: 0 auto; }}
  .glass {{
    background: var(--panel);
    backdrop-filter: blur(16px);
    border: 1px solid var(--line);
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
  }}
  .head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 12px; }}
  h1 {{
    font-size: 28px; font-weight: 700; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #ffffff, #cbd5e1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .updated {{ color: var(--muted); font-size: 14px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 20px; }}
  .stat {{ padding: 14px; text-align: center; }}
  .stat-k {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}
  .stat-v {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
  .chart-container {{ position: relative; width: 100%; height: 620px; margin-bottom: 24px; padding: 16px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; padding: 10px 14px; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 5px; font-size: 12px; font-weight: 500; }}
  .sw {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .section-title {{ color: #58a6ff; font-size: 1.1em; margin: 20px 0 8px; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ background: rgba(30,41,59,0.8); color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; padding: 8px 6px; text-align: left; border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 1; }}
  td {{ padding: 6px; border-bottom: 1px solid rgba(255,255,255,0.04); font-variant-numeric: tabular-nums; }}
  tr:hover {{ background: rgba(255,255,255,0.03); }}
  .highlight {{ background: rgba(34,197,94,0.1); }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 0.65em; font-weight: 600; margin-left: 4px; }}
  .badge-best {{ background: #238636; color: #fff; }}
  .badge-full {{ background: #1f6feb; color: #fff; }}
  .badge-quick {{ background: #484f58; color: #c9d1d9; }}
  .badge-remote {{ background: #6e40c9; color: #fff; }}
  .badge-local {{ background: #238636; color: #fff; }}
  .panel {{ padding: 16px; margin-bottom: 16px; }}
  .controls {{ display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; align-items: center; }}
  .controls label {{ font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 5px; }}
  .controls select, .controls input {{ background: rgba(255,255,255,0.05); border: 1px solid var(--line); color: var(--ink); border-radius: 6px; padding: 4px 8px; font: inherit; font-size: 12px; }}
  .controls button {{
    border: 1px solid var(--line); background: rgba(255,255,255,0.05); color: var(--ink);
    border-radius: 8px; padding: 6px 14px; font: inherit; font-weight: 500; cursor: pointer;
    transition: all 0.2s ease;
  }}
  .controls button:hover {{ background: rgba(255,255,255,0.1); transform: translateY(-1px); }}
  .table-wrap {{ max-height: 500px; overflow-y: auto; }}
  .note {{ color: #f59e0b; font-size: 11px; padding: 8px 12px; background: rgba(245,158,11,0.08); border-radius: 6px; margin: 8px 0; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <div>
      <h1>Unified Experiment Dashboard</h1>
      <div class="updated">Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")} | Total: {len(results)} experiments | Dataset: distinct5_512 (5 styles x 1000) unless noted</div>
    </div>
    <div class="controls">
      <label>Dataset:
        <select id="dataset-filter">
          <option value="all">All</option>
          <option value="distinct5_512" selected>distinct5_512</option>
          <option value="wikiart512">wikiart512</option>
        </select>
      </label>
      <label>Eval:
        <select id="eval-filter">
          <option value="all">All</option>
          <option value="full_eval">Full eval only</option>
          <option value="quick_eval">Quick eval only</option>
        </select>
      </label>
      <label>Family:
        <select id="family-filter">
          <option value="all">All</option>
          <option value="FC-SB">FC-SB</option>
          <option value="Spectral">Spectral</option>
          <option value="LANCET">LANCET</option>
          <option value="Baseline">Baseline</option>
          <option value="RW-other">RW-other</option>
        </select>
      </label>
      <button onclick="filterAndRedraw()">Apply</button>
    </div>
  </div>

  <div class="note">
    <strong>Note:</strong> quick_eval (n=6) scores are NOT comparable with full_eval scores. FC-SB/625 quick_eval clip_style ~0.82 vs full_eval ~0.73. Use "Full eval only" filter for fair comparison.
  </div>

  <div class="stats" id="stats-grid"></div>

  <section class="glass panel">
    <div class="section-title">Pareto: CLIP-Style vs Content Preservation (1-LPIPS)</div>
    <div class="legend" id="chart-legend"></div>
    <div class="chart-container">
      <canvas id="paretoChart"></canvas>
    </div>
  </section>

  <section class="glass panel">
    <div class="section-title">Best per Method (prefers full_eval)</div>
    <div class="table-wrap">
      <table id="best-table">
        <thead>
          <tr><th>#</th><th>Method</th><th>Experiment</th><th>Ep</th>
              <th>CLIP-S</th><th>1-LPIPS</th><th>delta_idt</th><th>CLIP-T</th><th>Eval</th><th>DS</th><th>Src</th></tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <section class="glass panel">
    <div class="section-title">All Results (searchable)</div>
    <div class="controls">
      <label>Search: <input type="text" id="search-input" placeholder="Filter by name..." oninput="filterTable()"></label>
      <label>Min CLIP-S: <input type="number" id="min-clip" step="0.01" value="0.60" style="width:70px" onchange="filterTable()"></label>
    </div>
    <div class="table-wrap" style="max-height: 700px;">
      <table id="all-table">
        <thead>
          <tr><th>Method</th><th>Experiment</th><th>Ep</th>
              <th>CLIP-S</th><th>1-LPIPS</th><th>delta_idt</th><th>CLIP-T</th><th>Eval</th><th>DS</th></tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </section>
</div>

<script>
const ALL_DATA = {chart_json};
const TABLE_DATA = {table_json};
const BEST_DATA = {best_json};
const METHOD_COLORS = {colors_json};
const METHODS = {methods_list};
const IDT_FLOOR = {idt_floor};
const BASELINES = {baselines_json};
const FCSB_BEST = {fcsb_best_json};

// ---- Stats ----
function renderStats() {{
  const grid = document.getElementById('stats-grid');
  const d5 = TABLE_DATA.filter(d => d.dataset === 'distinct5_512');
  const d5Full = d5.filter(d => d.eval_protocol === 'full_eval');
  const bestFull = d5Full.length > 0 ? d5Full.reduce((a,b) => a.clip_style > b.clip_style ? a : b) : null;
  const bestPareto = d5Full.length > 0 ? d5Full.reduce((a,b) => {{
    const sa = a.clip_style + (a.one_minus_lpips||0);
    const sb = b.clip_style + (b.one_minus_lpips||0);
    return sa > sb ? a : b;
  }}) : null;
  const fcsbFull = d5Full.filter(d => d.family === 'FC-SB');
  const bestFCSB = fcsbFull.length > 0 ? fcsbFull.reduce((a,b) => a.clip_style > b.clip_style ? a : b) : null;
  
  const stats = [
    {{ k: 'Total Experiments', v: TABLE_DATA.length, c: '#58a6ff' }},
    {{ k: 'distinct5 full_eval', v: d5Full.length, c: '#a78bfa' }},
    {{ k: 'Best Full clip_style', v: bestFull ? bestFull.clip_style.toFixed(4) : 'N/A', c: '#22c55e' }},
    {{ k: 'Best Full Method', v: bestFull ? bestFull.method : 'N/A', c: '#f59e0b' }},
    {{ k: 'Best FC-SB full', v: bestFCSB ? bestFCSB.clip_style.toFixed(4) : 'N/A', c: '#F59E0B' }},
    {{ k: 'Pareto Optimal', v: bestPareto ? (bestPareto.clip_style + (bestPareto.one_minus_lpips||0)).toFixed(4) : 'N/A', c: '#ec4899' }},
    {{ k: 'IDT Floor', v: IDT_FLOOR.toFixed(4), c: '#8E63C0' }},
    {{ k: 'Methods', v: [...new Set(TABLE_DATA.map(d => d.method))].length, c: '#64748B' }},
  ];
  grid.innerHTML = stats.map(s => `
    <div class="glass stat"><div class="stat-k">${{s.k}}</div><div class="stat-v" style="color:${{s.c}}">${{s.v}}</div></div>
  `).join('');
}}

// ---- Chart ----
let chart = null;
function renderChart(data) {{
  const datasets = [];
  const methodSet = [...new Set(data.map(d => d.method))];
  
  methodSet.forEach(method => {{
    const points = data.filter(d => d.method === method);
    const isQuick = points.some(p => p.eval_protocol === 'quick_eval');
    const baseColor = METHOD_COLORS[method] || '#94a3b8';
    
    datasets.push({{
      label: method,
      data: points.map(p => ({{ x: p.clip_style, y: p.one_minus_lpips, ...p }})),
      backgroundColor: isQuick ? baseColor + '40' : baseColor + 'cc',
      borderColor: baseColor,
      borderWidth: isQuick ? 0.5 : 1.5,
      pointRadius: isQuick ? 3 : 6,
      pointHoverRadius: isQuick ? 5 : 9,
      pointStyle: isQuick ? 'circle' : 'circle',
    }});
  }});
  
  if (chart) chart.destroy();
  const ctx = document.getElementById('paretoChart').getContext('2d');
  chart = new Chart(ctx, {{
    type: 'scatter',
    data: {{ datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      scales: {{
        x: {{
          title: {{ display: true, text: 'CLIP Style (higher = more style)', color: '#c9d1d9', font: {{ size: 12 }} }},
          min: 0.58, max: 0.85,
          grid: {{ color: '#21262d' }},
          ticks: {{ color: '#8b949e' }}
        }},
        y: {{
          title: {{ display: true, text: '1 - LPIPS (higher = better content)', color: '#c9d1d9', font: {{ size: 12 }} }},
          min: 0.10, max: 0.76,
          grid: {{ color: '#21262d' }},
          ticks: {{ color: '#8b949e' }}
        }}
      }},
      plugins: {{
        legend: {{ labels: {{ color: '#c9d1d9', usePointStyle: true, pointStyle: 'circle', font: {{ size: 10 }}, boxWidth: 8 }} }},
        tooltip: {{
          backgroundColor: '#161b22ee', titleColor: '#58a6ff', bodyColor: '#c9d1d9',
          borderColor: '#30363d', borderWidth: 1,
          callbacks: {{
            title: (items) => items[0].raw.name + (items[0].raw.sub_exp ? '/' + items[0].raw.sub_exp : ''),
            label: (item) => {{
              const d = item.raw;
              return [
                `Method: ${{d.method}} [${{d.eval_protocol}}]`,
                `Epoch: ${{d.epoch ?? 'N/A'}}`,
                `CLIP Style: ${{d.x?.toFixed(4)}}`,
                `1-LPIPS: ${{d.y?.toFixed(4)}}`,
                `delta_idt: ${{d.clip_s_delta_idt?.toFixed(4) ?? 'N/A'}}`,
                `Dataset: ${{d.dataset}}`,
              ].filter(Boolean);
            }}
          }}
        }}
      }}
    }},
    plugins: [{{
      id: 'referenceLines',
      afterDraw(chart) {{
        const ctx = chart.ctx;
        const xAxis = chart.scales.x;
        const yAxis = chart.scales.y;
        
        // IDT floor vertical line
        const xPos = xAxis.getPixelForValue(IDT_FLOOR);
        ctx.save();
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = '#8E63C0';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(xPos, yAxis.top);
        ctx.lineTo(xPos, yAxis.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#8E63C0';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.fillText('IDT floor ' + IDT_FLOOR, xPos + 3, yAxis.top + 12);
        ctx.restore();
        
        // Baseline markers
        Object.entries(BASELINES).forEach(([name, bl]) => {{
          if (bl.clip_style && bl.one_minus_lpips) {{
            const px = xAxis.getPixelForValue(bl.clip_style);
            const py = yAxis.getPixelForValue(bl.one_minus_lpips);
            ctx.save();
            ctx.fillStyle = bl.color || '#64748B';
            ctx.beginPath();
            ctx.moveTo(px, py - 8);
            ctx.lineTo(px + 6, py);
            ctx.lineTo(px, py + 8);
            ctx.lineTo(px - 6, py);
            ctx.closePath();
            ctx.fill();
            ctx.font = '9px -apple-system, sans-serif';
            ctx.fillText(name, px + 8, py + 3);
            ctx.restore();
          }} else if (bl.clip_style) {{
            const px = xAxis.getPixelForValue(bl.clip_style);
            ctx.save();
            ctx.setLineDash([3, 3]);
            ctx.strokeStyle = bl.color || '#64748B';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(px, yAxis.top);
            ctx.lineTo(px, yAxis.bottom);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = bl.color || '#64748B';
            ctx.font = '9px -apple-system, sans-serif';
            ctx.fillText(name, px + 3, yAxis.top + 24);
            ctx.restore();
          }}
        }});
        
        // FC-SB best markers
        Object.entries(FCSB_BEST).forEach(([name, bl]) => {{
          const px = xAxis.getPixelForValue(bl.clip_style);
          const py = yAxis.getPixelForValue(bl.one_minus_lpips);
          ctx.save();
          ctx.fillStyle = bl.color || '#F59E0B';
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(px, py, 8, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.font = 'bold 9px -apple-system, sans-serif';
          ctx.fillStyle = '#F59E0B';
          ctx.fillText(name, px + 10, py - 4);
          ctx.restore();
        }});
      }}
    }}]
  }});
  
  // Legend
  const legendEl = document.getElementById('chart-legend');
  const families = [...new Set(data.map(d => d.family))];
  const famColors = {{ 'FC-SB': '#F59E0B', 'Spectral': '#5BC0EB', 'LANCET': '#D64045', 'Baseline': '#64748B', 'Other': '#94a3b8' }};
  legendEl.innerHTML = families.map(f => 
    `<span><i class="sw" style="background:${{famColors[f] || '#94a3b8'}}"></i>${{f}}</span>`
  ).join('') + '<span style="color:#8b949e">| hollow=quick_eval, solid=full_eval</span>';
}}

// ---- Best Table ----
function renderBestTable() {{
  const tbody = document.querySelector('#best-table tbody');
  const sorted = Object.values(BEST_DATA).sort((a, b) => b.clip_style - a.clip_style);
  tbody.innerHTML = sorted.map((d, i) => {{
    const isBest = i === 0;
    const evalBadge = d.eval_protocol === 'full_eval' ? '<span class="badge badge-full">FULL</span>' : 
                      d.eval_protocol === 'quick_eval' ? '<span class="badge badge-quick">QUICK</span>' : '';
    const srcBadge = d.source === 'remote_cached' || d.source === 'remote_csv' ? '<span class="badge badge-remote">R</span>' :
                     d.source === 'local' ? '<span class="badge badge-local">L</span>' : '';
    return `<tr ${{isBest ? 'class="highlight"' : ''}}>
      <td>${{i + 1}}</td>
      <td>${{d.method}}</td>
      <td>${{d.name}}${{d.sub_exp ? '/' + d.sub_exp : ''}} ${{isBest ? '<span class="badge badge-best">BEST</span>' : ''}}</td>
      <td>${{d.epoch ?? '-'}}</td>
      <td>${{d.clip_style?.toFixed(4) ?? '-'}}</td>
      <td>${{d.one_minus_lpips?.toFixed(4) ?? '-'}}</td>
      <td>${{d.clip_s_delta_idt?.toFixed(4) ?? '-'}}</td>
      <td>${{d.clip_t?.toFixed(4) ?? '-'}}</td>
      <td>${{evalBadge}}</td>
      <td>${{d.dataset || '-'}}</td>
      <td>${{srcBadge}}</td>
    </tr>`;
  }}).join('');
}}

// ---- All Table ----
function renderAllTable(data) {{
  const tbody = document.querySelector('#all-table tbody');
  const sorted = [...data].sort((a, b) => b.clip_style - a.clip_style);
  tbody.innerHTML = sorted.slice(0, 500).map(d => {{
    const evalBadge = d.eval_protocol === 'full_eval' ? '<span class="badge badge-full">FULL</span>' : 
                      d.eval_protocol === 'quick_eval' ? '<span class="badge badge-quick">QUICK</span>' : '';
    return `<tr>
      <td>${{d.method}}</td>
      <td>${{d.name}}${{d.sub_exp ? '/' + d.sub_exp : ''}}</td>
      <td>${{d.epoch ?? '-'}}</td>
      <td>${{d.clip_style?.toFixed(4) ?? '-'}}</td>
      <td>${{d.one_minus_lpips?.toFixed(4) ?? '-'}}</td>
      <td>${{d.clip_s_delta_idt?.toFixed(4) ?? '-'}}</td>
      <td>${{d.clip_t?.toFixed(4) ?? '-'}}</td>
      <td>${{evalBadge}} ${{d.eval_type || ''}}</td>
      <td>${{d.dataset || '-'}}</td>
    </tr>`;
  }}).join('');
}}

// ---- Filters ----
function filterAndRedraw() {{
  const dsFilter = document.getElementById('dataset-filter').value;
  const evalFilter = document.getElementById('eval-filter').value;
  const famFilter = document.getElementById('family-filter').value;
  let filtered = [...ALL_DATA];
  let filteredTable = [...TABLE_DATA];
  if (dsFilter === 'distinct5_512') {{
    filtered = filtered.filter(d => d.dataset === 'distinct5_512');
    filteredTable = filteredTable.filter(d => d.dataset === 'distinct5_512');
  }} else if (dsFilter === 'wikiart512') {{
    filtered = filtered.filter(d => d.dataset === 'wikiart512');
    filteredTable = filteredTable.filter(d => d.dataset === 'wikiart512');
  }}
  if (evalFilter !== 'all') {{
    filtered = filtered.filter(d => d.eval_protocol === evalFilter);
    filteredTable = filteredTable.filter(d => d.eval_protocol === evalFilter);
  }}
  if (famFilter !== 'all') {{
    filtered = filtered.filter(d => d.family === famFilter);
    filteredTable = filteredTable.filter(d => d.family === famFilter);
  }}
  renderChart(filtered);
  renderAllTable(filteredTable);
}}

function filterTable() {{
  const search = document.getElementById('search-input').value.toLowerCase();
  const minClip = parseFloat(document.getElementById('min-clip').value) || 0;
  let data = TABLE_DATA.filter(d => d.clip_style >= minClip);
  if (search) data = data.filter(d => (d.name + d.method + d.sub_exp).toLowerCase().includes(search));
  renderAllTable(data);
}}

// ---- INIT ----
renderStats();
renderChart(ALL_DATA);
renderBestTable();
renderAllTable(TABLE_DATA);
</script>
</body>
</html>'''
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Dashboard saved to {path}")

# ---- MAIN ----
def main():
    do_remote = '--remote' in sys.argv
    csv_only = '--csv-only' in sys.argv
    
    all_results = []
    
    # 1. Scan local experiments
    if not csv_only:
        print("Scanning local experiments...")
        local_dirs = [
            (os.path.join(LOCAL_EXP, '625_fc_sb'), '625_fc_sb'),
            (os.path.join(LOCAL_EXP, 'fc_sb_r2'), 'fc_sb_r2'),
            (os.path.join(LOCAL_EXP, 'p3_remote_10h'), 'p3_remote_10h'),
            (os.path.join(LOCAL_EXP, 'tuning_deepdive'), 'tuning_deepdive'),
        ]
        # Also scan 620_spectral and 628_ablation locally if they exist
        if os.path.isdir(LOCAL_EXP):
            for d in sorted(os.listdir(LOCAL_EXP)):
                if d.startswith('620_spectral') or d.startswith('628_ablation') or d.startswith('p4_fusion'):
                    local_dirs.append((os.path.join(LOCAL_EXP, d), d))
        
        for target_dir, group_name in local_dirs:
            r = scan_local_dir(target_dir, group_name)
            all_results.extend(r)
            if r:
                print(f"  {group_name}: {len(r)} results")
        
        # Also read existing CSV data
        existing_csv = os.path.join(os.path.dirname(__file__), 'exp_all_results.csv')
        if os.path.isfile(existing_csv):
            print("Reading existing CSV...")
            with open(existing_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cs = row.get('clip_style', '')
                    if cs:
                        try:
                            cs_val = float(cs)
                            if cs_val > 0:
                                m = dict(row)
                                m['clip_style'] = cs_val
                                for k in ['content_lpips', 'one_minus_lpips', 'clip_s_delta_idt', 'clip_t', 'clip_content']:
                                    if m.get(k):
                                        try: m[k] = float(m[k])
                                        except: m[k] = None
                                    else:
                                        m[k] = None
                                try: m['epoch'] = int(m['epoch']) if m.get('epoch') else None
                                except: m['epoch'] = None
                                m['source'] = m.get('source', 'existing_csv')
                                m['fid'] = None
                                m['artfid'] = None
                                all_results.append(m)
                        except:
                            pass
    
    # 2. Scan remote
    if do_remote:
        print("Scanning remote server...")
        remote_results = scan_remote()
        print(f"Remote: {len(remote_results)} results")
        all_results.extend(remote_results)
    
    # 2a-2. Read baseline_metrics_unified.csv (Related Works baselines)
    rw_csv = os.path.join(LOCAL_BASE, 'baseline_metrics_unified.csv')
    if os.path.isfile(rw_csv):
        print("Reading baseline_metrics_unified.csv...")
        with open(rw_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            added = 0
            for row in reader:
                method = row.get('method', row.get('baseline', ''))
                cs = row.get('clip_style', '')
                if not cs:
                    continue
                try:
                    cs_val = float(cs)
                    if cs_val <= 0:
                        continue
                except:
                    continue
                
                baseline_id = row.get('baseline', '')
                source = row.get('source', '')
                
                group = f'RW/{source}'
                lpips = float(row['content_lpips']) if row.get('content_lpips') else None
                oml = round(1 - lpips, 4) if lpips else None
                clip_content = float(row['clip_content']) if row.get('clip_content') else None
                fid = float(row['fid']) if row.get('fid') else None
                artfid = float(row['art_fid']) if row.get('art_fid') else None
                
                m = {
                    'group': group,
                    'exp_dir': baseline_id,
                    'sub_exp': method,
                    'epoch': None,
                    'clip_style': cs_val,
                    'clip_s_delta_idt': cs_val - 0.6399 if not baseline_id.startswith('ours') else None,
                    'clip_t': float(row['clip_dir']) if row.get('clip_dir') else None,
                    'content_lpips': lpips,
                    'one_minus_lpips': oml,
                    'clip_content': clip_content,
                    'fid': fid,
                    'artfid': artfid,
                    'eval_type': row.get('protocol', ''),
                    'source': 'related_works',
                    'dataset': 'distinct5_512',
                }
                all_results.append(m)
                added += 1
        print(f"  Added {added} from RW baselines")
    
    # 2a-3. Read unified_eval_results.json (local unified reeval)
    unified_json = os.path.join(LOCAL_EXP, 'baseline_reeval', 'unified_eval_results.json')
    if os.path.isfile(unified_json):
        print("Reading unified_eval_results.json...")
        with open(unified_json, 'r', encoding='utf-8') as f:
            unified_data = json.load(f)
        added = 0
        for key, val in unified_data.items():
            # Read clip_t from summary.json if available
            clip_t = None
            summary_path = os.path.join(LOCAL_EXP, 'baseline_reeval', key, 'summary.json')
            if os.path.isfile(summary_path):
                try:
                    with open(summary_path, 'r', encoding='utf-8') as sf:
                        sdata = json.load(sf)
                    overview = sdata.get('analysis', {}).get('all_pairs_overview', {})
                    if overview and overview.get('clip_t') is not None:
                        clip_t = overview['clip_t']
                except:
                    pass
            
            m = {
                'group': 'RW/unified_reeval',
                'exp_dir': key,
                'sub_exp': val.get('method', key),
                'epoch': None,
                'clip_style': val.get('clip_style'),
                'clip_s_delta_idt': val.get('clip_s_delta_idt'),
                'clip_t': clip_t,
                'content_lpips': val.get('content_lpips'),
                'one_minus_lpips': val.get('one_minus_lpips'),
                'clip_content': 0.0,
                'fid': None,
                'artfid': None,
                'eval_type': 'unified_reeval',
                'source': 'local_4070',
                'dataset': 'distinct5_512',
            }
            all_results.append(m)
            added += 1
        print(f"  Added {added} from unified_eval_results.json")
    
    # Also try to read previously downloaded remote CSV
    remote_csv = os.path.join(os.path.dirname(__file__), 'exp_all_results_remote.csv')
    if os.path.isfile(remote_csv):
        print("Reading cached remote CSV...")
        with open(remote_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cs = row.get('clip_style', '')
                if cs:
                    try:
                        cs_val = float(cs)
                        if cs_val > 0:
                            m = dict(row)
                            m['clip_style'] = cs_val
                            for k in ['content_lpips', 'one_minus_lpips', 'clip_s_delta_idt', 'clip_t', 'clip_content', 'fid', 'artfid']:
                                if m.get(k):
                                    try: m[k] = float(m[k])
                                    except: m[k] = None
                                else:
                                    m[k] = None
                            try: m['epoch'] = int(m['epoch']) if m.get('epoch') else None
                            except: m['epoch'] = None
                            m['source'] = 'remote_cached'
                            all_results.append(m)
                    except:
                        pass
    
    # 2b. Read fig_distinct5_all_points_big.csv (manuscript figure data)
    fig5_csv = os.path.join(LOCAL_BASE, 'aaai2027', 'fig_distinct5_all_points_big.csv')
    if os.path.isfile(fig5_csv):
        print("Reading fig_distinct5_all_points_big.csv...")
        with open(fig5_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            added = 0
            for row in reader:
                cs = row.get('clip_style', '')
                if cs:
                    try:
                        cs_val = float(cs)
                        if cs_val > 0:
                            family = row.get('family', row.get('superfamily', 'unknown'))
                            m = {
                                'group': f'fig5/{family}',
                                'exp_dir': row.get('label', row.get('point_id', '')),
                                'sub_exp': row.get('variant', ''),
                                'epoch': None,
                                'clip_style': cs_val,
                                'clip_s_delta_idt': float(row['style_minus_idt']) if row.get('style_minus_idt') else None,
                                'clip_t': None,
                                'content_lpips': float(row['content_lpips']) if row.get('content_lpips') else None,
                                'one_minus_lpips': float(row['one_minus_lpips']) if row.get('one_minus_lpips') else None,
                                'clip_content': None,
                                'fid': None,
                                'artfid': None,
                                'eval_type': 'fig5_transfer',
                                'source': 'fig_distinct5_csv',
                                'dataset': 'distinct5_512',
                            }
                            all_results.append(m)
                            added += 1
                    except:
                        pass
        print(f"  Added {added} from fig5 CSV")
    
    # 2c. Add manuscript operating points and SaMAM diag data
    supplement = [
        {'group': 'samam', 'exp_dir': 'samam_diag_step2250', 'sub_exp': '', 'epoch': 2250,
         'clip_style': 0.581097, 'clip_s_delta_idt': None, 'clip_t': None,
         'content_lpips': 0.353820, 'one_minus_lpips': 0.646180, 'clip_content': 0.864654,
         'fid': 148.206, 'artfid': 148.206, 'eval_type': 'full_eval_pixel_space',
         'source': 'remote_metrics_csv_aggregate', 'dataset': 'distinct5_512'},
        {'group': 'samam', 'exp_dir': 'samam_diag_step2500', 'sub_exp': '', 'epoch': 2500,
         'clip_style': 0.582324, 'clip_s_delta_idt': None, 'clip_t': None,
         'content_lpips': 0.347108, 'one_minus_lpips': 0.652892, 'clip_content': 0.870103,
         'fid': None, 'artfid': None, 'eval_type': 'full_eval_pixel_space',
         'source': 'remote_metrics_csv_aggregate', 'dataset': 'distinct5_512'},
        {'group': 'samam', 'exp_dir': 'samam_diag_step3000', 'sub_exp': '', 'epoch': 3000,
         'clip_style': 0.588972, 'clip_s_delta_idt': None, 'clip_t': None,
         'content_lpips': 0.352127, 'one_minus_lpips': 0.647873, 'clip_content': 0.870294,
         'fid': 306.16, 'artfid': 345.60, 'eval_type': 'full_eval_pixel_space',
         'source': 'remote_metrics_csv_aggregate', 'dataset': 'distinct5_512'},
        {'group': 'manuscript', 'exp_dir': 'IDT_floor', 'sub_exp': '', 'epoch': None,
         'clip_style': 0.6399, 'clip_s_delta_idt': 0.0, 'clip_t': None, 'content_lpips': None,
         'one_minus_lpips': None, 'clip_content': None, 'fid': None, 'artfid': None,
         'eval_type': 'idt_transfer', 'source': 'manuscript_ledger', 'dataset': 'distinct5_512'},
        {'group': 'manuscript', 'exp_dir': 'SaMAM-2250_manuscript', 'sub_exp': '', 'epoch': 2250,
         'clip_style': 0.5523, 'clip_s_delta_idt': -0.0876, 'clip_t': None, 'content_lpips': 0.3605,
         'one_minus_lpips': 0.6395, 'clip_content': None, 'fid': 148.206, 'artfid': 148.206,
         'eval_type': 'manuscript_validated', 'source': 'manuscript_ledger', 'dataset': 'distinct5_512'},
    ]
    all_results.extend(supplement)
    print(f"  Added {len(supplement)} supplement records")
    
    # 3. Merge & dedup
    print(f"\nTotal before merge: {len(all_results)}")
    merged = merge_results(all_results, [])
    print(f"After dedup: {len(merged)}")
    
    # 4. Write CSV
    write_csv(merged, LOCAL_RESULTS_CSV)
    
    # 5. Generate HTML
    generate_html_dashboard(merged, LOCAL_DASHBOARD)
    
    # Summary
    print("\n=== SUMMARY ===")
    group_counts = Counter(r.get('group', 'unknown') for r in merged)
    for g, c in group_counts.most_common(15):
        group_results = [r for r in merged if r.get('group') == g and isinstance(r.get('clip_style'), (int, float))]
        if group_results:
            best = max(group_results, key=lambda x: x['clip_style'])
            print(f"  {g}: {c} results, best clip_style={best['clip_style']:.4f}, lpips={best.get('content_lpips')}, exp={best.get('exp_dir')}")

if __name__ == '__main__':
    main()
