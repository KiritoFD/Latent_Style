import json, os, glob

results = []
for d in glob.glob(r'I:/Github/Latent_Style/exp/inmortal-exp/phase2_*'):
    for f in glob.glob(os.path.join(d, '**', 'summary.json'), recursive=True):
        try:
            data = json.load(open(f))
            a = data.get('analysis', {}).get('all_pairs_overview', {})
            s = a.get('clip_style', 0)
            l = a.get('content_lpips', 0)
            if s and s > 0:
                name = os.path.basename(os.path.dirname(os.path.dirname(f)))
                results.append((name, s, l))
        except:
            pass

results.sort(key=lambda x: -x[1])
for n, s, l in results:
    print(f'{n}: style={s:.4f} lpips={l:.4f}')
