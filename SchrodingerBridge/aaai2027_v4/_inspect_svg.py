import re, html
with open('framework_sfm_main_ORIG.svg','r',encoding='utf-8') as f:
    txt=f.read()

# Find mxCell values for LL/LH/HL/HH
for label in ['LL','LH','HL','HH']:
    pattern = r'value="([^"]*' + label + r'[^"]*)".*?&lt;mxGeometry[^/]*x="([^"]+)"[^/]*y="([^"]+)"[^/]*width="([^"]+)"[^/]*height="([^"]+)"'
    m = re.search(pattern, txt, re.S)
    if m:
        val = html.unescape(m.group(1)).replace('&#xa;',' ')
        print(label, 'value:', val[:60])
        print('  geom x,y,w,h:', m.group(2), m.group(3), m.group(4), m.group(5))
    else:
        print(label, 'not found')
