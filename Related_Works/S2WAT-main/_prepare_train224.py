import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

src_content = Path(r"G:/GitHub/Latent_Style/style_data/train/photo")
style_roots = [
    Path(r"G:/GitHub/Latent_Style/style_data/train/cezanne"),
    Path(r"G:/GitHub/Latent_Style/style_data/train/Hayao"),
    Path(r"G:/GitHub/Latent_Style/style_data/train/monet"),
    Path(r"G:/GitHub/Latent_Style/style_data/train/vangogh"),
]
out_content = Path(r"G:/GitHub/Latent_Style/Related_Works/S2WAT-main/input/Train224/Content")
out_style = Path(r"G:/GitHub/Latent_Style/Related_Works/S2WAT-main/input/Train224/Style")
out_content.mkdir(parents=True, exist_ok=True)
out_style.mkdir(parents=True, exist_ok=True)

# Keep official preprocess behavior.
transform = T.Compose([
    T.Resize(512),
    T.RandomCrop((224, 224)),
])


def process_folder(src_dir: Path, out_dir: Path, prefix: str = ""):
    files = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}])
    n = 0
    for p in files:
        try:
            img = Image.open(p).convert('RGB')
            img = transform(img)
            stem = p.stem
            name = f"{prefix}{stem}.png"
            img.save(out_dir / name)
            n += 1
        except Exception:
            pass
    return n

n_content = process_folder(src_content, out_content)
n_style = 0
for r in style_roots:
    n_style += process_folder(r, out_style, prefix=f"{r.name}_")

print(f"processed_content={n_content}")
print(f"processed_style={n_style}")
