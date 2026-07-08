"""Download CLIP-ViT-H on remote server via modelscope."""
from modelscope import snapshot_download
path = snapshot_download('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', cache_dir='I:/modelscope_cache')
print(f'Done: {path}')
