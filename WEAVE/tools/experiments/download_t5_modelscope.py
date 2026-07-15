from modelscope.hub.file_download import model_file_download
import os

model_id = 'google/t5-v1_1-large'
cache_dir = '/mnt/i/Github/Latent_Style/eval_cache/modelscope'

files_to_download = [
    'pytorch_model.bin',
    'config.json',
    'spiece.model',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'generation_config.json'
]

print("Starting custom ModelScope downloads (PyTorch files only)...")
for f in files_to_download:
    print(f"Downloading {f}...")
    try:
        path = model_file_download(
            model_id=model_id,
            file_path=f,
            cache_dir=cache_dir
        )
        print(f" -> Saved to {path}")
    except Exception as e:
        print(f"Error downloading {f}: {e}")

print("All downloads complete!")
