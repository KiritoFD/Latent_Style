import re
from pathlib import Path

p = Path('g:/GitHub/Latent_Style/SchrodingerBridge/docs/630/aaai_arch_diagram_v6.drawio')
s = p.read_text(encoding='utf-8')
# Convert \( ... \) to $$ ... $$
s = re.sub(r'\\\\\(([^\\\n]+?)\\\\\)', r'$$\1$$', s)
p.write_text(s, encoding='utf-8')
print('converted inline delimiters to $$')
