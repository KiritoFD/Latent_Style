import os
from PIL import Image

# 支持的图片后缀
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

root = "/mnt/g/GitHub/Latent_Style/style_data/train/vangogh"  # 改成你的目录

for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in EXTS:
            continue

        src = os.path.join(dirpath, fn)
        base = os.path.splitext(fn)[0]
        dst = os.path.join(dirpath, f"{base}_flip{ext}")

        # 已存在就跳过（避免重复生成）
        if os.path.exists(dst):
            continue

        try:
            with Image.open(src) as im:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                # JPEG 统一转 RGB，避免保存报错
                if ext in {".jpg", ".jpeg"} and im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                im.save(dst)
                print("Saved:", dst)
        except Exception as e:
            print("Failed:", src, "->", e)
