import numpy as np

data = {
    'Ours e7':  {'lpips': 0.449, 'clip_s': 0.7165, 'params': 3.9, 'ec': 0.395, 'label': 'Ours e7'},
    'Ours e8':  {'lpips': 0.451, 'clip_s': 0.7158, 'params': 3.9, 'ec': 0.393, 'label': 'Ours e8'},
    'SaMST':    {'lpips': 0.466, 'clip_s': 0.7195, 'params': 6.0, 'ec': 0.384, 'label': 'SaMST'},
    'S2WAT':    {'lpips': 0.526, 'clip_s': 0.714, 'params': 65,  'ec': 0.338, 'label': 'S2WAT'},
    'AdaIN':    {'lpips': 0.630, 'clip_s': 0.713, 'params': 5,   'ec': 0.264, 'label': 'AdaIN'},
    'StyleID':  {'lpips': 0.750, 'clip_s': 0.760, 'params': 30,  'ec': 0.190, 'label': 'StyleID'},
    'CAST':     {'lpips': 0.726, 'clip_s': 0.665, 'params': 7.0, 'ec': 0.182, 'label': 'CAST'},
}

# Transform x: 1 - LPIPS so higher = better
for d in data.values():
    d['content'] = 1 - d['lpips']

# SVG dimensions and margins
svg_width = 600
svg_height = 500
left_margin = 80
right_margin = 50
top_margin = 50
bottom_margin = 80

plot_width = svg_width - left_margin - right_margin
plot_height = svg_height - top_margin - bottom_margin

# Data ranges
x_min, x_max = 0.20, 0.60
y_min, y_max = 0.60, 0.78

# Start building SVG
svg_lines = [
    f'<?xml version="1.0" encoding="UTF-8"?>',
    f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">',
    f'<defs>',
    f'<style>',
    f'.data-point {{ cursor: pointer; }}',
    f'.label {{ cursor: move; user-select: none; }}',
    f'</style>',
    f'</defs>',
    f'<!-- Background -->',
    f'<rect x="0" y="0" width="{svg_width}" height="{svg_height}" fill="white"/>',
    f'<!-- Plot area -->',
    f'<rect x="{left_margin}" y="{top_margin}" width="{plot_width}" height="{plot_height}" fill="white" stroke="black" stroke-width="0.5"/>',
]

# Add grid
grid_color = 'lightgray'
for x in np.linspace(x_min, x_max, 9):
    px = left_margin + (x - x_min) / (x_max - x_min) * plot_width
    svg_lines.append(f'<line x1="{px:.1f}" y1="{top_margin:.1f}" x2="{px:.1f}" y2="{top_margin + plot_height:.1f}" stroke="{grid_color}" stroke-width="0.5" opacity="0.5"/>')

for y in np.linspace(y_min, y_max, 10):
    py = top_margin + plot_height - (y - y_min) / (y_max - y_min) * plot_height
    svg_lines.append(f'<line x1="{left_margin:.1f}" y1="{py:.1f}" x2="{left_margin + plot_width:.1f}" y2="{py:.1f}" stroke="{grid_color}" stroke-width="0.5" opacity="0.5"/>')

# Colors for each method
colors_map = {
    'Ours e7': '#E74C3C', 'Ours e8': '#E74C3C', 'SaMST': '#F39C12', 
    'S2WAT': '#8E44AD', 'AdaIN': '#2ECC71', 'StyleID': '#3498DB', 'CAST': '#95A5A6'
}

# Add data points
for name, d in data.items():
    px = left_margin + (d['content'] - x_min) / (x_max - x_min) * plot_width
    py = top_margin + plot_height - (d['clip_s'] - y_min) / (y_max - y_min) * plot_height
    
    ms = (d['params'] if d['params'] else 20) * 2
    ms = max(8, min(ms, 30))
    
    svg_lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{ms:.1f}" fill="{colors_map[name]}" stroke="black" stroke-width="1" class="data-point {name}"/>')

# Text offsets
offsets = {
    'Ours e7': (20, -5),
    'Ours e8': (15, 12),
    'SaMST': (-40, -8),
    'S2WAT': (20, 12),
    'AdaIN': (-45, -5),
    'StyleID': (-50, -8),
    'CAST': (-40, 12),
}

# Add text labels
for name, d in data.items():
    px = left_margin + (d['content'] - x_min) / (x_max - x_min) * plot_width
    py = top_margin + plot_height - (d['clip_s'] - y_min) / (y_max - y_min) * plot_height
    
    off = offsets.get(name, (15, 10))
    text_x = px + off[0]
    text_y = py + off[1]
    
    font_weight = 'bold' if name in {'SaMST', 'Ours e7', 'Ours e8'} else 'normal'
    
    svg_lines.append(f'<text x="{text_x:.1f}" y="{text_y:.1f}" font-size="11px" font-family="Arial" fill="black" font-weight="{font_weight}" text-anchor="middle" class="label {name}">{d.get("label", name)}</text>')

# Add axes
svg_lines.append(f'<!-- X axis -->');
svg_lines.append(f'<line x1="{left_margin:.1f}" y1="{top_margin + plot_height:.1f}" x2="{left_margin + plot_width:.1f}" y2="{top_margin + plot_height:.1f}" stroke="black" stroke-width="1"/>')

svg_lines.append(f'<!-- Y axis -->');
svg_lines.append(f'<line x1="{left_margin:.1f}" y1="{top_margin:.1f}" x2="{left_margin:.1f}" y2="{top_margin + plot_height:.1f}" stroke="black" stroke-width="1"/>')

# X axis ticks and labels
x_ticks = np.linspace(x_min, x_max, 9)
for x in x_ticks:
    px = left_margin + (x - x_min) / (x_max - x_min) * plot_width
    svg_lines.append(f'<line x1="{px:.1f}" y1="{top_margin + plot_height:.1f}" x2="{px:.1f}" y2="{top_margin + plot_height + 5:.1f}" stroke="black" stroke-width="0.5"/>')
    svg_lines.append(f'<text x="{px:.1f}" y="{top_margin + plot_height + 20:.1f}" font-size="9px" font-family="Arial" text-anchor="middle">{x:.2f}</text>')

# Y axis ticks and labels
y_ticks = np.linspace(y_min, y_max, 10)
for y in y_ticks:
    py = top_margin + plot_height - (y - y_min) / (y_max - y_min) * plot_height
    svg_lines.append(f'<line x1="{left_margin - 5:.1f}" y1="{py:.1f}" x2="{left_margin:.1f}" y2="{py:.1f}" stroke="black" stroke-width="0.5"/>')
    svg_lines.append(f'<text x="{left_margin - 10:.1f}" y="{py + 3:.1f}" font-size="9px" font-family="Arial" text-anchor="end">{y:.3f}</text>')

# Axis labels
svg_lines.append(f'<text x="{left_margin + plot_width/2:.1f}" y="{svg_height - 20:.1f}" font-size="11px" font-family="Arial" font-weight="bold" text-anchor="middle">Content preservation (1-LPIPS) →</text>')

# Y-axis label with rotation
y_label_x = 20
y_label_y = top_margin + plot_height/2
svg_lines.append(f'<text x="{y_label_x:.1f}" y="{y_label_y:.1f}" font-size="11px" font-family="Arial" font-weight="bold" text-anchor="middle" transform="rotate(-90 {y_label_x:.1f} {y_label_y:.1f})">CLIP-style ↑</text>')

svg_lines.append('</svg>')

# Write SVG file
with open('fig_pareto_editable.svg', 'w', encoding='utf-8') as f:
    f.write('\n'.join(svg_lines))

print("Saved editable SVG: fig_pareto_editable.svg")

