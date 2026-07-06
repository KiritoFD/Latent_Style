from pathlib import Path

root = Path('F:/wikiart/wikiart')
styles = []
for d in sorted(root.iterdir()):
    if d.is_dir():
        n = len([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}])
        styles.append((d.name, n))
print(f'Total styles: {len(styles)}')
for name, n in styles:
    print(f'{name}: {n}')
