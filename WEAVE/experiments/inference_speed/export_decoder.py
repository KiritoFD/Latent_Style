"""
Export the VAE decoder (only) to ONNX for fixed-shape input.

ROOT-CAUSE FIX (previous version blew up to 22G VRAM):
  * previous script ran torch.jit.trace AND torch.onnx.export back-to-back in
    one process -> two full traced graphs + their activations resident at once.
  * NEITHER was wrapped in no_grad/inference_mode -> autograd graph + saved
    activations for the whole bs=6 decoder (64->512 upsampling) => huge VRAM.

This version:
  * ONLY exports ONNX (no jit in the same process).
  * Wraps EVERYTHING in torch.inference_mode() (no autograd graph).
  * Exports with a SMALL fixed batch (default 1); the static graph batch does
    not need to match the runtime decode batch when we use ORT.
  * Frees the model before the onnx.checker step.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

# Windows console defaults to GBK, which cannot encode the emoji (✅) that
# torch 2.11's dynamo ONNX exporter prints in verbose output -> UnicodeEncodeError.
# Force UTF-8 for std streams before anything is printed.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import VAEDecodeWrapper, download_vae_with_fallback  # noqa: E402

DEVICE = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-batch", type=int, default=6,
                    help="fixed export batch; ORT runtime batch must match (dynamic_axes=None)")
    ap.add_argument("--output-dir", default="experiments/inference_speed/exported")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    C = args.export_batch

    with torch.inference_mode():
        vae = download_vae_with_fallback("ema", device=DEVICE)
        vae = vae.to(device=DEVICE, dtype=torch.float16, memory_format=torch.channels_last)
        vae.eval()
        decoder = VAEDecodeWrapper(vae).to(device=DEVICE, dtype=torch.float16,
                                       memory_format=torch.channels_last)
        decoder.eval()
        scale = float(vae.config.scaling_factor)
        print(f"[loaded DECODER only] scaling_factor={scale} export_batch={C}", flush=True)

        dummy = torch.randn(C, 4, 64, 64, device=DEVICE, dtype=torch.float16)
        print(f"[mem before export] {torch.cuda.memory_allocated()/1e9:.2f} GB alloc", flush=True)

        onnx_path = out_dir / "decoder.onnx"
        # onnxscript is now installed -> use the default dynamo exporter (no dynamo=False).
        torch.onnx.export(
            decoder,
            dummy,
            str(onnx_path),
            input_names=["latent"],
            output_names=["image"],
            dynamic_axes=None,       # FIXED shape
            opset_version=18,
            do_constant_folding=True,
        )
        print(f"[mem after export]  {torch.cuda.memory_allocated()/1e9:.2f} GB alloc", flush=True)
        print(f"[export] onnx -> {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)", flush=True)

        # The dynamo exporter may externalize weights (0.64 MB file, weights lost).
        # Re-save with save_as_external_data=False to fuse ALL weights into the
        # single .onnx file so ORT/TRT can load it standalone.
        import onnx as _onnx

        _m = _onnx.load(str(onnx_path))
        _onnx.save_model(
            _m,
            str(onnx_path),
            save_as_external_data=False,
            all_tensors_to_one_file=True,
        )
        print(f"[export] re-saved with inline weights -> "
              f"{onnx_path.stat().st_size/1e6:.1f} MB", flush=True)

    # free GPU before the (CPU-side) checker
    del decoder, vae, dummy
    gc.collect()
    torch.cuda.empty_cache()

    import onnx
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    print(f"[verify] onnx OK "
          f"in={[i.name for i in m.graph.input]} out={[o.name for o in m.graph.output]}",
          flush=True)
    print("[done] onnx export complete, GPU released on exit.", flush=True)


if __name__ == "__main__":
    main()
