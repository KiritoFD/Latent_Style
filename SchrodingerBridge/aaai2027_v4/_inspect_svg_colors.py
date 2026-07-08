import re, sys, os

in_path = r"F:\aaai_arch_diagram_v16_staggered_bundle.drawio.svg"
out_path = r"F:\aaai_arch_diagram_v16_staggered_bundle.drawio_light.svg"

with open(in_path, 'r', encoding='utf-8') as f:
    txt = f.read()

print('length', len(txt))

color_set = set()
for pat in [r'fill="([^"]+)"', r'fill=([^"\s;]+)', r'background="([^"]+)"', r'stroke="([^"]+)"', r'color="([^"]+)"']:
    for c in re.findall(pat, txt):
        color_set.add(c)

print('found colors count', len(color_set))
for c in sorted(color_set)[:100]:
    print('  ', c)

# find dark colors
for c in sorted(color_set):
    lc = c.lower().strip()
    if lc in ['black', '#000000', '#000', '#111111', '#111', '#222222', '#222', '#121212', '#1e1e1e', '#333333', '#333']:
        print('DARK color', c)
    if lc in ['white', '#ffffff', '#fff']:
        print('LIGHT color', c)

# print style snippet with background
for m in re.finditer(r'background[^>]*>', txt[:5000]):
    print('background snippet:', txt[max(0, m.start()-50):m.end()+20])
