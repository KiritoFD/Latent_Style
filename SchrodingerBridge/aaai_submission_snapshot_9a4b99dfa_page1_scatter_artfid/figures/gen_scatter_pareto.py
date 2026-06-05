import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

data = {
    'Ours e7':  {'lpips': 0.449, 'clip_s': 0.7165, 'params': 3.9, 'ec': 0.395, 'label': 'Ours e7'},
    'Ours e8':  {'lpips': 0.451, 'clip_s': 0.7158, 'params': 3.9, 'ec': 0.393, 'label': 'Ours e8'},
    'SaMST':    {'lpips': 0.466, 'clip_s': 0.7195, 'params': 6.0, 'ec': 0.384, 'label': 'SaMST'},
    'S2WAT':    {'lpips': 0.526, 'clip_s': 0.714, 'params': 65,  'ec': 0.338},
    'AdaIN':    {'lpips': 0.630, 'clip_s': 0.713, 'params': 5,   'ec': 0.264},
    'StyleID':  {'lpips': 0.750, 'clip_s': 0.760, 'params': 30,  'ec': 0.190},
    'CAST':     {'lpips': 0.726, 'clip_s': 0.665, 'params': 7.0, 'ec': 0.182},
}

fig, ax = plt.subplots(figsize=(4.0, 3.5))

# Transform x: 1 - LPIPS so higher = better
for d in data.values():
    d['content'] = 1 - d['lpips']

ax.set_xlabel(r'Content preservation (1$-$LPIPS) $\uparrow$', fontsize=10)
ax.set_ylabel(r'CLIP-style $\uparrow$', fontsize=10)
ax.tick_params(labelsize=9)

colors = {'Ours e7': '#E74C3C', 'Ours e8': '#E74C3C', 'SaMST': '#F39C12', 
          'S2WAT': '#8E44AD', 'AdaIN': '#2ECC71', 'StyleID': '#3498DB', 'CAST': '#95A5A6'}
# Fine-tuned offsets to avoid overlapping: each point gets unique placement
offsets = {
    'Ours e7': (20, 8),       # Top right
    'Ours e8': (15, -15),     # Bottom right
    'SaMST': (-30, 12),       # Top left  
    'S2WAT': (8, -15),        # Bottom middle
    'AdaIN': (-35, 8),        # Left side
    'StyleID': (-40, 12),     # Far left top
    'CAST': (-30, -15),       # Bottom left
}

for name, d in data.items():
    ms = (d['params'] if d['params'] else 20) * 12
    ms = max(25, min(ms, 180))
    ax.scatter(d['content'], d['clip_s'], c=colors[name], s=ms,
               edgecolors='black', linewidth=0.5, zorder=5, alpha=0.9)
    
    # Label all methods
    off = offsets.get(name, (8, 8))
    label_text = d.get('label', name)
    # Make key methods bold
    fontweight = 'bold' if name in {'SaMST', 'Ours e7', 'Ours e8'} else 'normal'
    ax.annotate(label_text, (d['content'], d['clip_s']),
                textcoords='offset points', xytext=off,
                fontsize=8, alpha=0.9, fontweight=fontweight)

# Adjust axes: expand Y range so StyleID is at top, expand X for breathing room
ax.set_xlim(0.20, 0.60)
ax.set_ylim(0.60, 0.78)
ax.grid(True, alpha=0.15)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.3)
# Export as SVG (editable vector format) and PNG
plt.savefig('fig_pareto_lpips_vs_style.svg', format='svg', bbox_inches='tight')
plt.savefig('fig_pareto_lpips_vs_style.png', dpi=300, bbox_inches='tight')
print("Saved SVG and PNG")
