import re
from pathlib import Path
import xml.etree.ElementTree as ET

p = Path('g:/GitHub/Latent_Style/SchrodingerBridge/docs/630/aaai_arch_diagram_v6.drawio')
s = p.read_text(encoding='utf-8')

# Wrap any remaining single $...$ in $$...$$
s = re.sub(r'(?<!\$)\$([^$\n]+?)\$(?!\$)', r'$$\1$$', s)

root = ET.fromstring(s)
ns = {'mx': 'http://www.w3.org/1999/xhtml'}  # not used, mx has no namespace in file

for cell in root.iter('mxCell'):
    value = cell.get('value', '')
    style = cell.get('style', '')
    if '$$' in value and 'autosize' not in style:
        # Add autosize for shapes so MathJax-rendered labels fit
        if 'strokeColor=' in style or 'fillColor=' in style:
            style += 'autosize=1;'
            cell.set('style', style)

p.write_text(ET.tostring(root, encoding='unicode'), encoding='utf-8')
print('prepared drawio for GUI export')
