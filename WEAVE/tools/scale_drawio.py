import re
from pathlib import Path

p = Path('g:/GitHub/Latent_Style/SchrodingerBridge/docs/630/aaai_arch_diagram_v6.drawio')
s = p.read_text(encoding='utf-8')
scale = 1.5

def scale_attr(m):
    name = m.group(1)
    val = float(m.group(2))
    return f'{name}="{val * scale:g}"'

# Scale geometry numeric attributes
s = re.sub(r'\b(x|y|width|height)="([0-9.]+)"', scale_attr, s)
# Scale page size
s = re.sub(r'pageWidth="([0-9.]+)"', lambda m: f'pageWidth="{float(m.group(1))*scale:g}"', s)
s = re.sub(r'pageHeight="([0-9.]+)"', lambda m: f'pageHeight="{float(m.group(1))*scale:g}"', s)
# Scale dx/dy
s = re.sub(r'dx="([0-9.]+)"', lambda m: f'dx="{float(m.group(1))*scale:g}"', s)
s = re.sub(r'dy="([0-9.]+)"', lambda m: f'dy="{float(m.group(1))*scale:g}"', s)

p.write_text(s, encoding='utf-8')
print(f'scaled diagram by {scale}')
