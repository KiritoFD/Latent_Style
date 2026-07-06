import subprocess, os, numpy as np
from PIL import Image

CANDIDATES = [
    '2013-11-25 10_46_18',
    '2013-11-18 06_58_36',
    '2013-11-12 16_58_40',
    '2013-11-21 17_44_44',
    '2013-11-08 16_45_24',
]

def metrics(path):
    img = Image.open(path).convert('RGB')
    img_np = np.array(img)
    gray = np.array(img.convert('L'))
    rg = np.abs(img_np[:,:,0] - img_np[:,:,1])
    yb = np.abs(0.5*(img_np[:,:,0]+img_np[:,:,1]) - img_np[:,:,2])
    colorful = np.sqrt(rg.std()**2 + yb.std()**2) + 0.3*np.sqrt(rg.mean()**2+yb.mean()**2)
    from scipy import ndimage
    sharp = ndimage.laplace(gray).var()
    contrast = gray.std()
    sat = np.array(img.convert('HSV'))[:,:,1].mean()
    return {'colorful': colorful, 'sharp': sharp, 'contrast': contrast, 'saturation': sat}

# Download identity photos for all candidates
for c in CANDIDATES:
    src = f"photo_{c}_to_photo.jpg"
    local = f"_ident_{c.replace(' ','_').replace(':','')}.jpg"
    if os.path.exists(local):
        continue
    subprocess.run([
        'scp', '-P', '2222', '-o', 'LogLevel=ERROR',
        f'administrator@100.115.18.62:I:/exp_256_photo2art/identity_256/images/{src}',
        local
    ], capture_output=True)

print('=== Output + content metrics for candidates ===')
for c in CANDIDATES:
    out = f"_pv_photo_{c.replace(' ','_')}.png"
    content = f"_ident_{c.replace(' ','_').replace(':','')}.jpg"
    if not os.path.exists(out):
        continue
    m_out = metrics(out)
    m_content = metrics(content) if os.path.exists(content) else {k:0 for k in m_out}
    print(f"\n{c}")
    print(f"  OUTPUT  colorful={m_out['colorful']:.1f} sharp={m_out['sharp']:.0f} contrast={m_out['contrast']:.1f} sat={m_out['saturation']:.1f}")
    print(f"  CONTENT colorful={m_content['colorful']:.1f} sharp={m_content['sharp']:.0f} contrast={m_content['contrast']:.1f} sat={m_content['saturation']:.1f}")
