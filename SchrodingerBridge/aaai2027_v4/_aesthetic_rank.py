import csv, os, math, numpy as np
from PIL import Image

def colorfulness(img_rgb):
    # Hasler & Süsstrunk metric
    rg = np.abs(img_rgb[:,:,0] - img_rgb[:,:,1])
    yb = np.abs(0.5 * (img_rgb[:,:,0] + img_rgb[:,:,1]) - img_rgb[:,:,2])
    return np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)

def laplacian_var(img_gray):
    from scipy import ndimage
    lap = ndimage.laplace(img_gray)
    return lap.var()

def contrast(img_gray):
    return img_gray.std()

def entropy(img_gray):
    hist, _ = np.histogram(img_gray, bins=256, range=(0,256), density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def saturation(img_rgb):
    hsv = np.array(img_rgb.convert('HSV'))
    return hsv[:,:,1].mean()

# Read metrics
rows = []
with open('_metrics_e10.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['src_style'] == 'photo' and row['tgt_style'] == 'vangogh':
            src = row['src_image']
            fname = f"_pv_photo_{src.replace(' ', '_').replace('.jpg', '')}.png"
            if not os.path.exists(fname):
                continue
            img = Image.open(fname).convert('RGB')
            img_np = np.array(img)
            gray = np.array(img.convert('L'))
            rows.append({
                'src': src,
                'clip_s': float(row['clip_style']),
                'clip_c': float(row['clip_content']),
                'lpips': float(row['content_lpips']),
                'colorful': colorfulness(img_np),
                'sharp': laplacian_var(gray),
                'contrast': contrast(gray),
                'entropy': entropy(gray),
                'saturation': saturation(img),
            })

# Normalize and score
for r in rows:
    r['norm_s'] = (r['clip_s'] - 0.6) / (0.77 - 0.6)
    r['norm_c'] = (r['clip_c'] - 0) / 0.001  # all zero, skip
    r['norm_lpips'] = max(0, 1 - abs(r['lpips'] - 0.25) / 0.25)
    r['norm_color'] = (r['colorful'] - 10) / 60
    r['norm_sharp'] = (r['sharp'] - 20) / 200
    r['norm_contrast'] = (r['contrast'] - 30) / 50
    r['norm_entropy'] = (r['entropy'] - 5) / 3
    r['score'] = (r['norm_s'] * 0.35 + r['norm_lpips'] * 0.25 +
                  r['norm_color'] * 0.12 + r['norm_sharp'] * 0.12 +
                  r['norm_contrast'] * 0.08 + r['norm_entropy'] * 0.08)

rows.sort(key=lambda x: x['score'], reverse=True)

print('=== Aesthetic + metric composite ranking ===')
hdr = '{:<4} {:<25} {:>7} {:>7} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9}'.format(
    'Rank', 'Source', 'CLIP-S', 'LPIPS', 'Color', 'Sharp', 'Contrast', 'Entropy', 'Saturate', 'Score')
print(hdr)
print('-' * len(hdr))
for i, r in enumerate(rows):
    print('{:<4} {:<25} {:>7.4f} {:>7.4f} {:>7.1f} {:>9.1f} {:>9.1f} {:>9.2f} {:>9.2f} {:>9.4f}'.format(
        i+1, r['src'][:25], r['clip_s'], r['lpips'], r['colorful'], r['sharp'],
        r['contrast'], r['entropy'], r['saturation'], r['score']))
