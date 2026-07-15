import sys, json, torch
from pathlib import Path

# Load checkpoint config
ck = torch.load(r'I:\Github\Latent_Style\SchrodingerBridge\exp\712_sf1_subband\epoch_0005.pt', map_location='cpu', weights_only=False)
cfg = ck.get('config', {})
print('=== Checkpoint Config (key fields) ===')
print('training.test_image_dir:', cfg.get('training', {}).get('test_image_dir', 'NOT_SET'))
print('data.style_subdirs:', cfg.get('data', {}).get('style_subdirs', 'NOT_SET'))

# Apply override (same as eval)
override_path = r'I:\Github\Latent_Style\SchrodingerBridge\configs\_sf1_eval_override.json'
override = json.loads(Path(override_path).read_text())
print('\n=== Override ===')
print(json.dumps(override, indent=2))

# Merge override into cfg (shallow merge for model section)
for section, values in override.items():
    if section not in cfg:
        cfg[section] = {}
    cfg[section].update(values)

print('\n=== After override ===')
print('model.solver_family:', cfg.get('model', {}).get('solver_family', 'NOT_SET'))

# Resolve test_dir
test_dir_raw = cfg.get('training', {}).get('test_image_dir', '')
print('\n=== Test dir resolution ===')
print('test_dir_raw:', test_dir_raw)
test_dir = Path(test_dir_raw)
print('test_dir exists:', test_dir.exists())
print('test_dir is_absolute:', test_dir.is_absolute())

# Check style_subdirs
style_subdirs = cfg.get('data', {}).get('style_subdirs', [])
print('style_subdirs:', style_subdirs)

# Check each style subdir
test_images = {}
for style_id, style_name in enumerate(style_subdirs):
    s_dir = test_dir / style_name
    exists = s_dir.exists()
    images = []
    if exists:
        images = sorted([p for p in s_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.webp']])
    print(f'  style {style_id} ({style_name}): dir={s_dir}, exists={exists}, images={len(images)}')
    if exists:
        test_images[style_id] = (style_name, images)

print('\n=== Summary ===')
print('test_images keys:', list(test_images.keys()))
print('total images:', sum(len(v[1]) for v in test_images.values()))

# Check _prefer_readable_eval_image_root candidates
root_str = str(test_dir)
print('\n=== Fallback candidates ===')
if '_samam_512_classview' in root_str:
    alt1 = Path(root_str.replace('_samam_512_classview', '_512_images'))
    print(f'  alt1 (512_images): {alt1} exists={alt1.exists()}')
if '_classview' in root_str and '_classview_real' not in root_str:
    alt2 = Path(root_str.replace('_classview', '_classview_real'))
    print(f'  alt2 (classview_real): {alt2} exists={alt2.exists()}')
