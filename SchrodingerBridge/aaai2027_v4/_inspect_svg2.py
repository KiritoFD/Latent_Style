import re, html
with open('framework_sfm_main_ORIG.svg','r',encoding='utf-8') as f:
    txt=f.read()

# Find key text labels and their positions
labels = ['WCT','Style ID','Style Memory','z_t','Haar','Flow Matching','Velocity Network','Endpoint','ideal routed','Spectral']
for label in labels:
    idx=txt.find('>'+label+'<')
    if idx!=-1:
        snippet = txt[max(0,idx-400):idx+100]
        m = re.search(r'<text x="([^"]+)" y="([^"]+)"[^>]*>[^<]*'+label, snippet)
        if m:
            print(label, 'x=',m.group(1),'y=',m.group(2))
        else:
            print(label, 'found but no coords in nearby text')
    else:
        print(label, 'not found')
