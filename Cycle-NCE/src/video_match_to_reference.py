from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _channel_match(
    src: np.ndarray,
    ref: np.ndarray,
    strength: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 255.0,
) -> np.ndarray:
    s_mean = float(src.mean())
    s_std = float(src.std())
    r_mean = float(ref.mean())
    r_std = float(ref.std())
    scale = r_std / (s_std + 1e-6)
    matched = (src - s_mean) * scale + r_mean
    if strength < 1.0:
        matched = src * (1.0 - strength) + matched * strength
    return np.clip(matched, clip_min, clip_max)


def match_frame_to_ref(
    src_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    luma_strength: float,
    chroma_strength: float,
) -> np.ndarray:
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    out_lab = np.empty_like(src_lab)
    out_lab[..., 0] = _channel_match(src_lab[..., 0], ref_lab[..., 0], strength=luma_strength)
    out_lab[..., 1] = _channel_match(src_lab[..., 1], ref_lab[..., 1], strength=chroma_strength)
    out_lab[..., 2] = _channel_match(src_lab[..., 2], ref_lab[..., 2], strength=chroma_strength)

    out_bgr = cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out_bgr


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Match result video brightness/color/contrast to a reference video.")
    p.add_argument("--ref", type=Path, required=True, help="Reference/original video")
    p.add_argument("--src", type=Path, required=True, help="Stylized/source video to be corrected")
    p.add_argument("--out", type=Path, required=True, help="Output corrected video path")
    p.add_argument("--luma-strength", type=float, default=0.9, help="0~1, stronger brightness/contrast alignment")
    p.add_argument("--chroma-strength", type=float, default=0.7, help="0~1, stronger color alignment")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not args.ref.exists():
        raise FileNotFoundError(f"Reference video not found: {args.ref}")
    if not args.src.exists():
        raise FileNotFoundError(f"Source video not found: {args.src}")

    ref_cap = cv2.VideoCapture(str(args.ref))
    src_cap = cv2.VideoCapture(str(args.src))
    if not ref_cap.isOpened():
        raise RuntimeError(f"Failed to open reference video: {args.ref}")
    if not src_cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {args.src}")

    fps = src_cap.get(cv2.CAP_PROP_FPS) or ref_cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )

    count = 0
    try:
        while True:
            ok_s, src = src_cap.read()
            if not ok_s:
                break
            ok_r, ref = ref_cap.read()
            if not ok_r:
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_r, ref = ref_cap.read()
                if not ok_r:
                    ref = src
            if ref.shape[:2] != src.shape[:2]:
                ref = cv2.resize(ref, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LANCZOS4)

            out = match_frame_to_ref(
                src_bgr=src,
                ref_bgr=ref,
                luma_strength=float(args.luma_strength),
                chroma_strength=float(args.chroma_strength),
            )
            writer.write(out)
            count += 1
    finally:
        ref_cap.release()
        src_cap.release()
        writer.release()

    print(json.dumps({"ok": True, "out": str(args.out), "frames": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
