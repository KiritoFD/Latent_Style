import re, html
with open('framework_sfm_main_ORIG.svg','r',encoding='utf-8') as f:
    txt=f.read()

# Find the note about ideal routed limit
idx = txt.find('ideal routed')
if idx != -1:
    print('Found at index', idx)
    print(txt[max(0,idx-300):idx+500])
    print('---')
    # find surrounding text coords
    m = re.search(r'<text x="([^"]+)" y="([^"]+)"[^>]*>([^<]*(?:ideal|Note|routed)[^<]*)</text>', txt[max(0,idx-500):idx+500])
    if m:
        print('coords:', m.group(1), m.group(2), 'text:', m.group(3))
else:
    print('ideal routed not found')

# Also find WCT text
idx2 = txt.find('WCT')
print('\nWCT idx', idx2)
if idx2 != -1:
    print(txt[max(0,idx2-200):idx2+200])
