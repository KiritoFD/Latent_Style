
from pathlib import Path

W, H = 2160, 1560
bg = '#f8fafc'
ink = '#0f172a'
muted = '#475569'
muted2 = '#64748b'
line_main = '#2563eb'
line_cond = '#d97706'
line_train = '#0f766e'
line_gray = '#6b7280'
fill_main = '#eaf2ff'
stroke_main = '#3b82f6'
fill_cond = '#fff2e7'
stroke_cond = '#d97706'
fill_train = '#e8fbf6'
stroke_train = '#0f766e'
fill_opt = '#f8fafc'
stroke_opt = '#94a3b8'
panel_main = '#f2f7ff'
panel_cond = '#fff8f1'
panel_train = '#f3fffb'
parts = []

def add(s):
    parts.append(s)

def esc(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

def rect(x, y, w, h, rx=18, fill='#fff', stroke='#000', sw=2, dashed=None, opacity=None):
    attrs = [f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"', f'rx="{rx}"', f'fill="{fill}"', f'stroke="{stroke}"', f'stroke-width="{sw}"']
    if dashed:
        attrs.append(f'stroke-dasharray="{dashed}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    add(f"<rect {' '.join(attrs)}/>")

def circle(cx, cy, r, fill='#fff', stroke='#000', sw=2):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def line(x1, y1, x2, y2, color='#000', sw=1.8, marker='arrow-main', dashed=None):
    attrs = [f'x1="{x1}"', f'y1="{y1}"', f'x2="{x2}"', f'y2="{y2}"', f'stroke="{color}"', f'stroke-width="{sw}"', 'fill="none"', f'marker-end="url(#{marker})"']
    if dashed:
        attrs.append(f'stroke-dasharray="{dashed}"')
    add(f"<line {' '.join(attrs)}/>")

def poly(points, color='#000', sw=1.8, marker='arrow-main', dashed=None):
    pts = ' '.join(f'{x},{y}' for x, y in points)
    attrs = [f'points="{pts}"', 'fill="none"', f'stroke="{color}"', f'stroke-width="{sw}"', 'stroke-linecap="round"', 'stroke-linejoin="round"', f'marker-end="url(#{marker})"']
    if dashed:
        attrs.append(f'stroke-dasharray="{dashed}"')
    add(f"<polyline {' '.join(attrs)}/>")

def text_center(x, y, lines, cls='body', fill=ink, weight=None, line_gap=22):
    if isinstance(lines, str):
        lines = [lines]
    add(f'<text x="{x}" y="{y}" text-anchor="middle" class="{cls}" fill="{fill}"' + (f' font-weight="{weight}"' if weight else '') + '>')
    for i, t in enumerate(lines):
        dy = 0 if i == 0 else line_gap
        add(f'  <tspan x="{x}" dy="{dy}">{esc(t)}</tspan>')
    add('</text>')

def text_left(x, y, lines, cls='body', fill=ink, weight=None, line_gap=22):
    if isinstance(lines, str):
        lines = [lines]
    add(f'<text x="{x}" y="{y}" text-anchor="start" class="{cls}" fill="{fill}"' + (f' font-weight="{weight}"' if weight else '') + '>')
    for i, t in enumerate(lines):
        dy = 0 if i == 0 else line_gap
        add(f'  <tspan x="{x}" dy="{dy}">{esc(t)}</tspan>')
    add('</text>')

def stacked_box(x, y, w, h, title, lines, fill, stroke, stack=1, stack_dx=0, stack_dy=-10, dashed=None, subtitle=None, title_fill=ink):
    for i in range(stack - 1, 0, -1):
        rect(x + stack_dx * i, y + stack_dy * i, w, h, rx=18, fill=fill, stroke=stroke, sw=1.4, opacity=0.58)
    rect(x, y, w, h, rx=18, fill=fill, stroke=stroke, sw=2, dashed=dashed)
    text_center(x + w/2, y + 28, title, cls='label', fill=title_fill)
    base_y = y + 54
    if subtitle:
        text_center(x + w/2, base_y, subtitle, cls='tiny', fill=muted2)
        base_y += 22
    text_center(x + w/2, base_y, lines, cls='body', fill=muted)

def tag(x, y, w, h, text, fill, stroke, color=ink):
    rect(x, y, w, h, rx=13, fill=fill, stroke=stroke, sw=1.5)
    text_center(x + w/2, y + 24, text, cls='small', fill=color, weight='700')

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>')
add('<style>')
add('.title { font: 700 34px Arial, Helvetica, sans-serif; fill: #0f172a; }')
add('.subtitle { font: 15px Arial, Helvetica, sans-serif; fill: #475569; }')
add('.panel { font: 700 18px Arial, Helvetica, sans-serif; fill: #334155; }')
add('.label { font: 700 17px Arial, Helvetica, sans-serif; }')
add('.body { font: 14px Arial, Helvetica, sans-serif; }')
add('.small { font: 13px Arial, Helvetica, sans-serif; }')
add('.tiny { font: 12px Arial, Helvetica, sans-serif; }')
add('.caption { font: 12px Arial, Helvetica, sans-serif; fill: #64748b; }')
add('</style>')
for name, color in [('arrow-main', line_main), ('arrow-cond', line_cond), ('arrow-train', line_train), ('arrow-gray', line_gray)]:
    add(f'<marker id="{name}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 z" fill="{color}"/></marker>')
add('</defs>')
rect(0, 0, W, H, rx=0, fill=bg, stroke='none', sw=0)
rect(22, 22, W-44, H-44, rx=24, fill='#ffffff', stroke='#dbe3ef', sw=1.5)
text_center(W/2, 62, 'Latent AdaCUT — cleaned code-aligned architecture', cls='title')
text_center(W/2, 90, 'Transformer-style organization: orthogonal routing, no arrow-text overlap, and the current mainline highlighted', cls='subtitle')
left = (60, 120, 620, 1360)
center = (720, 120, 720, 1360)
right = (1480, 120, 620, 1360)
rect(*left, rx=26, fill=panel_cond, stroke='#f4d6b6', sw=1.5)
rect(*center, rx=26, fill=panel_main, stroke='#cfe0ff', sw=1.5)
rect(*right, rx=26, fill=panel_train, stroke='#c7eee4', sw=1.5)
text_left(left[0]+24, left[1]+34, 'Conditioning interfaces', cls='panel')
text_left(center[0]+24, center[1]+34, 'Main latent residual stylizer', cls='panel')
text_left(right[0]+24, right[1]+34, 'Legend and training objectives', cls='panel')
legend_x = 1520
legend_y = 182
tag(legend_x, legend_y, 170, 38, 'main path', fill_main, stroke_main)
tag(legend_x+190, legend_y, 170, 38, 'conditioning', fill_cond, stroke_cond)
tag(legend_x+380, legend_y, 170, 38, 'optional / id-only', fill_opt, stroke_opt)
text_left(legend_x, legend_y+64, ['Solid arrows: active mainline. Dashed arrows: fallback, ablation,', 'or currently gated-off paths.'], cls='small', fill=muted)
text_left(legend_x, legend_y+104, ['Paper-level takeaway: the bottleneck SemanticCrossAttn is the primary', 'style writer; the decoder adds texture; the color highway carries low-frequency appearance.'], cls='small', fill=muted)
style_id = (120, 178, 220, 72)
global_code = (120, 298, 360, 96)
ref_latent = (120, 536, 360, 82)
ref_proj = (120, 652, 360, 92)
prior_box = (120, 814, 360, 82)
spatial_map = (120, 930, 360, 86)
note_box = (120, 1084, 360, 94)
stacked_box(*style_id, 'Style ID s_t', ['discrete target domain'], fill_cond, stroke_cond)
stacked_box(*global_code, 'style_emb  →  global style code w', ['feeds DecoderTextureBlock', 'optional shallow / skip modulators exist in code'], fill_cond, stroke_cond)
stacked_box(*ref_latent, 'Reference style latent z_s', ['training and exemplar-style inference'], fill_cond, stroke_cond)
stacked_box(*ref_proj, 'Resize to 16×16  +  1×1 style_map_proj', ['projects reference latent into bottleneck channel space'], fill_cond, stroke_cond)
stacked_box(*prior_box, 'Learned style prior P(s_t)', ['style_spatial_id_16[s_t]', 'used for style-ID-only inference'], fill_cond, stroke_cond, dashed='7 6')
stacked_box(*spatial_map, 'Spatial style map M_s', ['either Proj(z_s) or P(s_t)', 'feeds the bottleneck painter'], fill_cond, stroke_cond)
stacked_box(*note_box, 'Current forward-pass caveat', ['shallow style blocks and skip routing remain implemented,', 'but the main forward path passes gate = 0 on those routes'], fill_opt, stroke_opt, dashed='8 6')
inp = (920, 170, 320, 76)
lift = (920, 282, 320, 84)
hires = (900, 402, 360, 98)
down = (920, 536, 320, 76)
body = (900, 648, 360, 118)
up = (920, 802, 320, 76)
skip = (920, 914, 320, 88)
decoder = (900, 1038, 360, 104)
dec_out = (920, 1178, 320, 72)
delta = (920, 1286, 320, 82)
out = (920, 1434, 320, 76)
color = (1286, 662, 110, 84)
stacked_box(*inp, 'Input latent z_c', ['4 × 32 × 32'], fill_main, stroke_main)
stacked_box(*lift, 'Lift stem', ['3×3 Conv + SiLU', '4 → lift_channels'], fill_main, stroke_main)
stacked_box(*hires, 'Hi-res content stem  ×  H', ['32×32 content features', 'CrossAttnAdaGN exists here but is not active in the current mainline'], fill_main, stroke_main, stack=2)
stacked_box(*down, 'Downsample', ['4×4 Conv, stride 2', 'to 16×16 body'], fill_main, stroke_main)
stacked_box(*body, 'SemanticCrossAttn body  ×  N', ['primary style injection site', 'content query × spatial style map key/value'], fill_main, stroke_main, stack=3)
stacked_box(*color, 'Color highway', ['bilinear upsample', '+ 1×1 conv'], fill_cond, stroke_cond)
stacked_box(*up, 'Upsample + blur', ['16×16 → 32×32'], fill_main, stroke_main)
stacked_box(*skip, 'Skip fusion', ['skip_squeeze + spatial dropout', 'with gate = 0, this behaves like a light content skip'], fill_main, stroke_main)
stacked_box(*decoder, 'DecoderTextureBlock  ×  M', ['writes local texture residuals using w', 'main decoder-side style control'], fill_main, stroke_main, stack=2)
stacked_box(*dec_out, 'dec_out', ['predicts Δ_raw in latent space'], fill_main, stroke_main)
stacked_box(*delta, 'Residual delta composer', ['Δz = Δ_raw  +  γ · Δ_color', 'step_size and style_strength scale this update in integrate()'], fill_cond, stroke_cond)
cx, cy, r = 1080, 1406, 22
circle(cx, cy, r, fill='#ffffff', stroke=stroke_main, sw=2)
text_center(cx, cy+6, '+', cls='label', fill=stroke_main)
stacked_box(*out, 'Output latent z_out', ['z_c  +  step · Δz'], fill_main, stroke_main)
text_left(1520, 338, 'Training-only losses used by the current objective', cls='panel')
text_left(1520, 366, ['The boxes below summarize what the implementation actually optimizes;', 'they are separated from the forward path so the main trunk stays readable.'], cls='small', fill=muted)
swd_box = (1540, 462, 500, 118)
color_box = (1540, 616, 500, 118)
idt_box = (1540, 770, 500, 134)
opt_box = (1540, 940, 500, 112)
stacked_box(*swd_box, 'Cross-domain SWD', ['unified SWD, or micro high-pass + macro low-pass SWD', 'computed on cross-style pairs'], fill_train, stroke_train)
stacked_box(*color_box, 'Color statistics loss', ['pseudo-RGB → YUV statistics', 'brightness / contrast / tint / saturation'], fill_train, stroke_train)
stacked_box(*idt_box, 'Identity family', ['self-style anchor  ||z_out − z_c||_1', 'topology alignment across styles', 'cross-style repel when output stays too close'], fill_train, stroke_train)
stacked_box(*opt_box, 'Optional regularizers and diagnostics', ['soft repulsive, auxiliary delta variance,', 'attention entropy / attention max'], fill_opt, stroke_opt, dashed='8 6')
main_cent = 1080
for (bx1, by1, bw1, bh1), (bx2, by2, bw2, bh2) in [(inp, lift), (lift, hires), (hires, down), (down, body), (body, up), (up, skip), (skip, decoder), (decoder, dec_out), (dec_out, delta)]:
    line(main_cent, by1+bh1, main_cent, by2, color=line_main, sw=1.7, marker='arrow-main')
line(main_cent, delta[1]+delta[3], cx, cy-r, color=line_main, sw=1.7, marker='arrow-main')
line(cx, cy+r, main_cent, out[1], color=line_main, sw=1.7, marker='arrow-main')
poly([(920, 208), (804, 208), (804, cy), (cx-r, cy)], color=line_main, sw=1.6, marker='arrow-main')
text_left(818, 1368, 'residual anchor z_c', cls='tiny', fill=muted2)
poly([(920, 451), (860, 451), (860, 958), (920, 958)], color=line_main, sw=1.6, marker='arrow-main')
text_left(750, 724, 'content skip path', cls='tiny', fill=muted2)
poly([(1260, 707), (1286, 707)], color=line_cond, sw=1.6, marker='arrow-cond')
poly([(1341, 746), (1341, 1327), (1240, 1327)], color=line_cond, sw=1.6, marker='arrow-cond')
text_left(1310, 1246, 'low-frequency appearance', cls='tiny', fill=muted2)
line(230, style_id[1]+style_id[3], 230, global_code[1], color=line_cond, sw=1.6, marker='arrow-cond')
line(300, ref_latent[1]+ref_latent[3], 300, ref_proj[1], color=line_cond, sw=1.6, marker='arrow-cond')
line(300, ref_proj[1]+ref_proj[3], 300, spatial_map[1], color=line_cond, sw=1.6, marker='arrow-cond')
poly([(120, 214), (96, 214), (96, 855), (120, 855)], color=line_cond, sw=1.5, marker='arrow-cond', dashed='7 6')
line(300, prior_box[1]+prior_box[3], 300, spatial_map[1], color=line_cond, sw=1.5, marker='arrow-cond', dashed='7 6')
text_left(106, 780, 'id-only fallback', cls='tiny', fill=muted2)
poly([(480, 973), (780, 973), (780, 707), (900, 707)], color=line_cond, sw=1.7, marker='arrow-cond')
text_left(792, 688, 'main appearance carrier', cls='tiny', fill=muted2)
poly([(480, 346), (764, 346), (764, 1090), (900, 1090)], color=line_cond, sw=1.7, marker='arrow-cond')
text_left(780, 1068, 'w → decoder texture control', cls='tiny', fill=muted2)
poly([(480, 334), (740, 334), (740, 450), (900, 450)], color=line_gray, sw=1.4, marker='arrow-gray', dashed='7 6')
poly([(480, 358), (720, 358), (720, 958), (920, 958)], color=line_gray, sw=1.4, marker='arrow-gray', dashed='7 6')
text_left(524, 1104, 'dashed: implemented but currently gated off in the main forward path', cls='tiny', fill=muted2)
poly([(1240, 1472), (1510, 1472), (1510, 522), (1540, 522)], color=line_train, sw=1.7, marker='arrow-train')
poly([(1510, 676), (1540, 676)], color=line_train, sw=1.7, marker='arrow-train')
poly([(1510, 837), (1540, 837)], color=line_train, sw=1.7, marker='arrow-train')
poly([(1510, 996), (1540, 996)], color=line_train, sw=1.5, marker='arrow-train', dashed='7 6')
text_left(1522, 1088, 'All loss boxes consume z_out; their comparison targets are z_s and/or z_c as listed inside each box.', cls='tiny', fill=muted2)
rect(90, 1498, 1980, 40, rx=12, fill='#fff7ed', stroke='#f4c28a', sw=1.2)
text_left(110, 1517, ['Recommended paper narrative: explicit dual conditioning (reference latent or learned style prior),', 'bottleneck semantic painter as the primary style writer, decoder-side texture refinement, color highway, and residual latent update.'], cls='small', fill=muted)
text_left(120, 1216, ['Reference-style conditioning is the default during training;', 'the learned prior is the fallback when no reference style latent is supplied.'], cls='small', fill=muted)
text_left(760, 152, ['Main flow is centered and purely vertical;', 'side routes use dedicated gutters so arrows never pass through labels.'], cls='small', fill=muted)
add('</svg>')
svg = '\n'.join(parts)
Path('./model.svg').write_text(svg, encoding='utf-8')
html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Latent AdaCUT cleaned architecture</title>
  <style>
    body {{ margin: 0; background: #f8fafc; font-family: Arial, Helvetica, sans-serif; color: #0f172a; }}
    .wrap {{ max-width: 2200px; margin: 0 auto; padding: 20px; }}
    .card {{ background: #fff; border: 1px solid #dbe3ef; border-radius: 20px; overflow: hidden; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06); }}
    .figure {{ width: 100%; display: block; }}
    .meta {{ padding: 14px 18px 20px; border-top: 1px solid #e2e8f0; color: #475569; font-size: 14px; line-height: 1.55; }}
    .meta b {{ color: #0f172a; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      {svg}
      <div class="meta"><b>Layout update:</b> the figure is now organized into conditioning, main model, and training-objective lanes. The main trunk is strictly vertical, all side connections use reserved gutters, dashed arrows denote fallback or currently gated-off routes, and no arrow crosses through text labels.</div>
    </div>
  </div>
</body>
</html>
'''