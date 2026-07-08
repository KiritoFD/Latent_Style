import re, html
with open('framework_sfm_main_ORIG.svg','r',encoding='utf-8') as f:
    txt=f.read()

# Find all rect elements and their positions
rects = re.findall(r'<rect[^>]*>', txt)
print('total rects', len(rects))

# Parse each rect's x,y,w,h,fill,stroke
parsed = []
for r in rects:
    attrs = {}
    for k in ['x','y','width','height','fill','stroke']:
        m = re.search(rf'{k}="([^"]+)"', r)
        attrs[k] = m.group(1) if m else None
    try:
        x,y,w,h = float(attrs['x']), float(attrs['y']), float(attrs['width']), float(attrs['height'])
        parsed.append((x,y,w,h,attrs['fill'],attrs['stroke']))
    except:
        pass

# Print rects in the subband area (x<400, y 250-450)
print('\nRects in subband area:')
for p in parsed:
    x,y,w,h,fill,stroke = p
    if x < 400 and 250 < y < 450:
        print(f'  x={x:6.1f} y={y:6.1f} w={w:6.1f} h={h:6.1f} fill={fill} stroke={stroke}')

# Print rects in right area (x>1000)
print('\nRects on right side:')
for p in parsed:
    x,y,w,h,fill,stroke = p
    if x > 1000 and y < 550:
        print(f'  x={x:6.1f} y={y:6.1f} w={w:6.1f} h={h:6.1f} fill={fill} stroke={stroke}')
