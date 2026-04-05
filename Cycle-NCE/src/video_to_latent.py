from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL


def _round_hw_to_8(width: int, height: int) -> tuple[int, int]:
    w = max(8, (width // 8) * 8)
    h = max(8, (height // 8) * 8)
    return w, h


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Encode video frames into VAE latents for fast inference.")
    p.add_argument("--input-video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Output latent file (.pt)")
    p.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--size", type=int, default=256, help="If >0, resize each frame to square size. If <=0, keep original rounded to /8.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not args.input_video.exists():
        raise FileNotFoundError(f"Input video not found: {args.input_video}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch_dtype = torch.float16 if args.dtype == "fp16" and device == "cuda" else torch.float32

    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch_dtype).to(device)
    vae.eval()
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))

    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.input_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if int(args.size) > 0:
        tw = int(args.size)
        th = int(args.size)
    else:
        tw, th = _round_hw_to_8(ow, oh)

    latents_list: list[torch.Tensor] = []
    frame_count = 0
    frames: list[np.ndarray] = []

    def _encode_batch(batch_bgr: list[np.ndarray]) -> None:
        arrs = []
        for bgr in batch_bgr:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if (rgb.shape[1], rgb.shape[0]) != (tw, th):
                rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            arr = rgb.astype(np.float32) / 255.0
            arr = arr * 2.0 - 1.0
            arrs.append(arr)
        x = np.stack(arrs, axis=0)
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device=device, dtype=torch_dtype)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mean * scaling_factor
        latents_list.append(z.cpu())

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            frame_count += 1
            if len(frames) >= max(1, int(args.batch_size)):
                _encode_batch(frames)
                frames.clear()
        if frames:
            _encode_batch(frames)
    finally:
        cap.release()

    if not latents_list:
        raise RuntimeError("No frames encoded from input video.")

    latents = torch.cat(latents_list, dim=0).contiguous()
    payload = {
        "latents": latents,  # [T,4,H/8,W/8]
        "fps": fps,
        "frame_count": int(frame_count),
        "orig_size": [ow, oh],
        "proc_size": [tw, th],
        "vae": str(args.vae),
        "scaling_factor": scaling_factor,
        "dtype": str(latents.dtype).replace("torch.", ""),
        "source_video": str(args.input_video),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps({k: v for k, v in payload.items() if k != "latents"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(args.output), "meta": str(meta_path), "frames": frame_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
