from pathlib import Path
from transformers import CLIPModel, CLIPProcessor

cache_dir = Path(r'F:\GitHub\Latent_Style\Cycle-NCE\eval_cache\hf')
model_id = 'openai/clip-vit-base-patch32'
print('Downloading', model_id, 'to', cache_dir, flush=True)
model = CLIPModel.from_pretrained(model_id, cache_dir=str(cache_dir), local_files_only=False)
processor = CLIPProcessor.from_pretrained(model_id, cache_dir=str(cache_dir), local_files_only=False)
print('Model class:', model.__class__.__name__, flush=True)
print('Processor class:', processor.__class__.__name__, flush=True)
root = cache_dir / 'models--openai--clip-vit-base-patch32'
print('Root exists:', root.exists(), flush=True)
interesting = {
    'config.json',
    'preprocessor_config.json',
    'tokenizer_config.json',
    'tokenizer.json',
    'vocab.json',
    'merges.txt',
    'special_tokens_map.json',
    'model.safetensors',
    'pytorch_model.bin'
}
for p in sorted(root.rglob('*')):
    if p.is_file() and p.name in interesting:
        print('FILE', p.relative_to(root), p.stat().st_size, flush=True)
