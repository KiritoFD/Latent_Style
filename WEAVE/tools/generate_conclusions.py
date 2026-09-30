"""Generate comprehensive baseline reproduction conclusions.

Reads:
- exp/baseline_v2/eval/unified_results.json (metrics)
- .trae/autoresearch/baseline_repro/state/findings.jsonl (timing data)

Outputs:
- exp/baseline_v2/baseline_conclusions.md (supporting conclusions document)
- exp/baseline_v2/baseline_summary_table.csv (machine-readable summary)
"""
import sys
import json
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2')
UNIFIED = ROOT / 'eval' / 'unified_results.json'
FINDINGS = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\findings.jsonl')
OUT_MD = ROOT / 'baseline_conclusions.md'
OUT_CSV = ROOT / 'baseline_summary_table.csv'


def load_findings():
    """Load timing data from findings.jsonl."""
    findings = {}
    if not FINDINGS.exists():
        print(f"  WARNING: {FINDINGS} not found")
        return findings
    with open(FINDINGS, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                method = entry.get('method', '')
                # For cut_*, aggregate into cut entry
                if method.startswith('cut_'):
                    if 'cut' not in findings:
                        findings['cut'] = {'train_min': 0, 'infer_min': 0, 'subfindings': []}
                    findings['cut']['train_min'] += entry.get('train_min', 0) or 0
                    findings['cut']['subfindings'].append(entry)
                elif method == 'cut':
                    # Summary entry for cut - update fields but preserve subfindings
                    if 'cut' not in findings:
                        findings['cut'] = {'train_min': 0, 'infer_min': 0, 'subfindings': []}
                    findings['cut']['train_min'] = entry.get('train_min', findings['cut'].get('train_min', 0)) or findings['cut'].get('train_min', 0)
                    findings['cut']['infer_min'] = entry.get('infer_min', findings['cut'].get('infer_min', 0)) or findings['cut'].get('infer_min', 0)
                    findings['cut']['clip_style'] = entry.get('clip_style')
                    findings['cut']['content_lpips'] = entry.get('content_lpips')
                    findings['cut']['clip_s_delta_idt'] = entry.get('clip_s_delta_idt')
                    findings['cut']['n_pairs'] = entry.get('n_pairs')
                    findings['cut']['n_styles'] = entry.get('n_styles', len(findings['cut'].get('subfindings', [])))
                    findings['cut']['note'] = entry.get('note', '')
                else:
                    findings[method] = entry
            except json.JSONDecodeError:
                continue
    # For CUT, compute average if subfindings exist
    if 'cut' in findings and findings['cut'].get('subfindings'):
        sub = findings['cut']['subfindings']
        if not findings['cut'].get('train_min'):
            train_times = [s.get('train_min', 0) or 0 for s in sub]
            findings['cut']['train_min'] = sum(train_times)
        findings['cut']['n_styles'] = len(sub)
    return findings


def load_metrics():
    """Load evaluation metrics from unified_results.json."""
    if not UNIFIED.exists():
        print(f"  WARNING: {UNIFIED} not found")
        return {}
    with open(UNIFIED, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_method_display(method):
    """Pretty-print method name."""
    display = {
        'identity': 'Identity (copy)',
        'adain': 'AdaIN',
        'sdedit_str0.10': 'SDEdit (s=0.10)',
        'sdedit_str0.20': 'SDEdit (s=0.20)',
        'sdedit_str0.35': 'SDEdit (s=0.35)',
        'sdedit_str0.40': 'SDEdit (s=0.40)',
        'sdturbo': 'SD-Turbo (1-step)',
        'styleid': 'StyleID',
        'samst': 'SaMST',
        'cut': 'CUT',
        'samam': 'SaMam',
    }
    return display.get(method, method)


def method_category(method):
    """Classify method by training requirement."""
    if method in ('identity',):
        return 'baseline'
    if method in ('adain', 'sdedit_str0.10', 'sdedit_str0.20', 'sdedit_str0.35',
                   'sdedit_str0.40', 'sdturbo', 'styleid'):
        return 'inference-only'
    if method in ('samst', 'cut', 'samam'):
        return 'training-required'
    return 'unknown'


def generate_conclusions():
    metrics = load_metrics()
    findings = load_findings()

    if not metrics:
        print("ERROR: No metrics found")
        return

    # Build summary table
    methods = sorted(metrics.keys(), key=lambda m: metrics[m].get('clip_style', 0))

    lines = []
    lines.append("# Baseline Reproduction Conclusions")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n## Evaluation Protocol")
    lines.append(f"- Dataset: distinct5_512 (5 styles × 30 test images = 150 source)")
    lines.append(f"- Pairs per method: 750 (150 source × 5 target styles)")
    lines.append(f"- Metrics: LPIPS-VGG (content fidelity), CLIP-ViT-L/14 style (style strength)")
    lines.append(f"- Identity baseline: CLIP-S=0.6933, LPIPS=0.0")
    lines.append(f"- Δ_idt = CLIP-S(method) - CLIP-S(identity)")
    lines.append(f"\n## Summary Table")
    lines.append("")
    lines.append("| Method | Category | Train (min) | Infer (min) | CLIP-S | LPIPS | Δ_idt | n_pairs |")
    lines.append("|--------|----------|-------------|-------------|--------|-------|-------|---------|")

    csv_rows = ["method,category,train_min,infer_min,clip_style,content_lpips,clip_s_delta_idt,n_pairs"]

    for m in methods:
        mt = metrics[m]
        cat = method_category(m)
        f = findings.get(m, {})
        train_min = f.get('train_min', 0) or 0
        infer_min = f.get('infer_min', 0) or 0
        clip_s = mt.get('clip_style', 'N/A')
        lpips = mt.get('content_lpips', 'N/A')
        delta = mt.get('clip_s_delta_idt', 'N/A')
        n = mt.get('n_pairs', 750)

        clip_s_str = f"{clip_s:.4f}" if isinstance(clip_s, (int, float)) else str(clip_s)
        lpips_str = f"{lpips:.4f}" if isinstance(lpips, (int, float)) else str(lpips)
        delta_str = f"{delta:+.4f}" if isinstance(delta, (int, float)) else str(delta)

        lines.append(f"| {format_method_display(m)} | {cat} | {train_min:.1f} | {infer_min:.1f} | {clip_s_str} | {lpips_str} | {delta_str} | {n} |")
        csv_rows.append(f"{m},{cat},{train_min},{infer_min},{clip_s},{lpips},{delta},{n}")

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")

    # Analysis
    if metrics:
        best_style = max(metrics.keys(), key=lambda m: metrics[m].get('clip_style', 0))
        best_content = min(metrics.keys(), key=lambda m: metrics[m].get('content_lpips', 999) if metrics[m].get('content_lpips') is not None else 999)

        lines.append(f"### Style Transfer Strength (CLIP-S, higher = more style)")
        lines.append(f"- Best: {format_method_display(best_style)} (CLIP-S={metrics[best_style].get('clip_style', 0):.4f})")
        lines.append(f"- Identity baseline: CLIP-S=0.6933")
        lines.append(f"- Methods below identity (negative Δ_idt):")
        for m in methods:
            delta = metrics[m].get('clip_s_delta_idt', 0)
            if delta is not None and delta < 0:
                lines.append(f"  - {format_method_display(m)}: Δ_idt={delta:+.4f}")
        lines.append("")

        lines.append(f"### Content Preservation (LPIPS, lower = better preservation)")
        lines.append(f"- Best: {format_method_display(best_content)} (LPIPS={metrics[best_content].get('content_lpips', 0):.4f})")
        lines.append(f"- Identity baseline: LPIPS=0.0")
        lines.append("")

        # Trade-off analysis
        lines.append("### Style-Content Trade-off")
        lines.append("")
        lines.append("| Tier | Methods | Characteristic |")
        lines.append("|------|---------|----------------|")

        tiers = {
            'High style, low content': [],
            'Balanced': [],
            'Low style, high content': [],
            'Near-identity': [],
        }
        for m in methods:
            clip_s = metrics[m].get('clip_style', 0) or 0
            lpips = metrics[m].get('content_lpips', 0) or 0
            delta = metrics[m].get('clip_s_delta_idt', 0) or 0
            if lpips < 0.05:
                tiers['Near-identity'].append(m)
            elif delta > 0.1 and lpips > 0.4:
                tiers['High style, low content'].append(m)
            elif delta > 0.05 and lpips < 0.4:
                tiers['Balanced'].append(m)
            else:
                tiers['Low style, high content'].append(m)

        for tier, ms in tiers.items():
            if ms:
                names = ', '.join(format_method_display(m) for m in ms)
                lines.append(f"| {tier} | {names} | |")

        lines.append("")

        # Training time analysis
        train_methods = {m: f for m, f in findings.items() if method_category(m) == 'training-required' and (f.get('train_min', 0) or 0) > 0}
        if train_methods:
            lines.append("### Training Time Comparison")
            lines.append("")
            lines.append("| Method | Train (min) | Train (hr) | Per-style (min) |")
            lines.append("|--------|-------------|------------|-----------------|")
            for m, f in sorted(train_methods.items(), key=lambda x: x[1].get('train_min', 0)):
                tm = f.get('train_min', 0) or 0
                n_styles = f.get('n_styles', 1)
                per_style = tm / n_styles if n_styles > 0 else 0
                lines.append(f"| {format_method_display(m)} | {tm:.1f} | {tm/60:.2f} | {per_style:.1f} |")
            lines.append("")

    lines.append("## Reproduction Notes")
    lines.append("")
    lines.append("### Methods Failed/Skipped")
    lines.append("- **S2WAT**: Failed (environment/dependency issues on remote)")
    lines.append("")
    lines.append("### Methods Pending")
    pending = []
    if 'cut' not in metrics:
        pending.append('CUT (evaluation in progress)')
    if 'samam' not in metrics:
        pending.append('SaMam (training + inference pending)')
    if pending:
        for p in pending:
            lines.append(f"- {p}")
    else:
        lines.append("- None (all methods evaluated)")
    lines.append("")

    content = '\n'.join(lines)
    OUT_MD.write_text(content, encoding='utf-8')
    print(f"Wrote {OUT_MD}")

    csv_content = '\n'.join(csv_rows)
    OUT_CSV.write_text(csv_content, encoding='utf-8')
    print(f"Wrote {OUT_CSV}")

    print(f"\nMethods in metrics: {len(metrics)}")
    print(f"Methods in findings: {len(findings)}")
    print("\n==CONCLUSIONS_DONE==")


if __name__ == '__main__':
    generate_conclusions()
