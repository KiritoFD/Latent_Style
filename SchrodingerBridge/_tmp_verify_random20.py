from pathlib import Path
import json

root = Path('G:/GitHub/Latent_Style/Dataset/wikiart_random20_512/wikiart_random20_512')
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))

for split_name in ['train', 'test']:
    split_dir = root / 'images' / split_name
    print(f'\n{split_name}:')
    for style, info in manifest['splits'][split_name].items():
        style_dir = split_dir / style
        if style_dir.exists():
            n = len([p for p in style_dir.iterdir() if p.is_file()])
        else:
            n = 0
        print(f'  {style}: manifest={info["count"]} actual={n}')
