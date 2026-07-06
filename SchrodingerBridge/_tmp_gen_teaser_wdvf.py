"""Generate WD-VF teaser images from the Random-20 checkpoint at 512x512."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.inference import LGTInference, load_vae, encode_image, decode_latent, tensor_to_pil

CHECKPOINT = Path("g:/GitHub/Latent_Style/SchrodingerBridge/exp/630_random20_heun_5ep/epoch_0005.pt")
OUT_DIR = Path("g:/GitHub/Latent_Style/SchrodingerBridge/aaai2027_v3/teaser_wdvf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Random-20 style_id mapping
STYLE_IDS = {
    "Early_Renaissance": 5,
    "Impressionism": 9,
    "Minimalism": 11,
    "Rococo": 16,
    "Ukiyo_e": 19,
}

# (source_image_path, target_style_name, output_name)
PAIRS = [
    (
        "g:/GitHub/Latent_Style/SchrodingerBridge/exp/baseline/images/identity/Early_Renaissance__Early_Renaissance__andrea-mantegna_maria-with-the-sleeping-child-1455__to__Ukiyo_e.png",
        "Ukiyo_e",
        "wdvf_er_to_ukiyo_e.png",
    ),
]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(device=str(device))
    inf = LGTInference(str(CHECKPOINT), device=str(device), num_steps=1)

    model_scale = float(getattr(inf.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)

    for src_path, tgt_style, out_name in PAIRS:
        img = Image.open(src_path).convert("RGB").resize((512, 512))
        tensor = torch.from_numpy(np.array(img)).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0

        z = encode_image(vae, tensor, device=str(device))
        if abs(scale_in - 1.0) > 1e-4:
            z = z * scale_in

        # Model runs in float32; VAE encode returns float16.
        z = z.float()
        z_out = inf.transfer_style(z, target_style_id=STYLE_IDS[tgt_style], num_steps=20)

        if abs(scale_out - 1.0) > 1e-4:
            z_out = z_out * scale_out
        out = decode_latent(vae, z_out, device=str(device))
        tensor_to_pil(out).save(OUT_DIR / out_name)
        print(f"Saved {OUT_DIR / out_name}")


if __name__ == "__main__":
    main()
