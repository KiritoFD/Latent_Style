import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

data = {
    'Ours':     {'lpips': 0.451, 'clip_s': 0.716, 'params': 3.9, 'ec': 0.393},
    'SaMST':    {'lpips': 0.466, 'clip_s': 0.719, 'params': 6.0, 'ec': 0.384},
    'S2WAT':    {'lpips': 0.526, 'clip_s': 0.714, 'params': 65,  'ec': 0.338},
    'AdaIN':    {'lpips': 0.630, 'clip_s': 0.713, 'params': 5,   'ec': 0.264},
    'StyleID':  {'lpips': 0.750, 'clip_s': 0.760, 'params': 30,  'ec': 0.190},
    'CAST':     {'lpips': 0.726, 'clip_s': 0.665, 'params': 7.0, 'ec': 0.182},
}

fig, ax = plt.subplots(figsize=(3.5, 3.0))

# Transform x: 1 - LPIPS so higher = better
for d in data.values():
    d['content'] = 1 - d['lpips']

ax.set_xlabel(r'Content preservation (1$-$LPIPS) $\uparrow$', fontsize=9)
ax.set_ylabel('CLIP-style $\uparrow$', fontsize=9)
ax.tick_params(labelsize=8)

colors = {'Ours': '#E74C3C', 'SaMST': '#F39C12', 'S2WAT': '#8E44AD',
          'AdaIN': '#2ECC71', 'StyleID': '#3498DB', 'CAST': '#95A5A6'}
offsets = {'Ours': (6,6), 'SaMST': (6,-10), 'S2WAT': (6,6),
           'AdaIN': (6,-8), 'StyleID': (6,6), 'CAST': (6,-8)}

for name, d in data.items():
    ms = (d['params'] if d['params'] else 20) * 12
    ms = max(25, min(ms, 180))
    ax.scatter(d['content'], d['clip_s'], c=colors[name], s=ms,
               edgecolors='black', linewidth=0.4, zorder=5, alpha=0.9)
    off = offsets.get(name, (6,6))
    ax.annotate(name, (d['content'], d['clip_s']),
                textcoords='offset points', xytext=off,
                fontsize=7, alpha=0.85)

ax.set_xlim(0.22, 0.57)
ax.set_ylim(0.64, 0.78)
ax.grid(True, alpha=0.15)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.3)
plt.savefig('fig_pareto_lpips_vs_style.png', dpi=300, bbox_inches='tight')
print("Saved")
