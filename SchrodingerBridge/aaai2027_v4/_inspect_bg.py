import re, sys, os

in_path = r"F:\aaai_arch_diagram_v16_staggered_bundle.drawio.svg"
out_svg = r"F:\aaai_arch_diagram_v16_staggered_bundle.drawio_light.svg"
out_png = r"g:\GitHub\Latent_Style\SchrodingerBridge\aaai2027_v4\framework_sfm_main.png"

with open(in_path, 'r', encoding='utf-8') as f:
    txt = f.read()

# --- diagnostic: find background info ---
m = re.search(r'<svg[^>]*background=[\"\']([^\"\']+)[\"\']', txt)
if m:
    print('svg background=', m.group(1))
for m in re.finditer(r'background-color:\s*([^;\s]+)', txt):
    print('background-color', m.group(1))
m = re.search(r'background=\"([^\"]+)\"', txt)
if m:
    print('mxGraphModel background=', m.group(1))
for color in ['#1F2937', '#111827', '#111111', '#000000']:
    idx = txt.find('fill=\"' + color + '\"')
    if idx != -1:
        snippet = txt[max(0, idx - 100): idx + 200]
        print('fill', color, 'snippet:', snippet[:300])
        break
