"""Quick smoke test for build_offline_dino_cache_blockmask.apply_block_mask."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools" / "experiments"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# Import the function under test
import importlib.util
spec = importlib.util.spec_from_file_location(
    "blockmask", _TOOLS_DIR / "build_offline_dino_cache_blockmask.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
apply_block_mask = mod.apply_block_mask


def main() -> int:
    # Create a synthetic 512x512 RGB image with all pixels = (128, 64, 200)
    arr = np.full((512, 512, 3), [128, 64, 200], dtype=np.uint8)
    img = Image.fromarray(arr)
    print(f"[smoke] input size: {img.size}, mean RGB: {np.array(img).mean(axis=(0,1))}")

    # Apply block mask: ratio=0.6, block_size=128, seed=42
    # Expectation: 4x4 grid = 16 blocks, 60% = 9 blocks blacked out
    # Black area = 9 * 128 * 128 = 147456 pixels out of 512*512 = 262144 (56.25% of pixels)
    rng = random.Random(42)
    masked = apply_block_mask(img, mask_ratio=0.6, block_size=128, rng=rng)
    masked_arr = np.array(masked)
    print(f"[smoke] masked size: {masked.size}")

    # Count black blocks
    black_mask = (masked_arr == 0).all(axis=-1)  # H, W
    black_pixel_count = int(black_mask.sum())
    total_pixels = 512 * 512
    black_ratio = black_pixel_count / total_pixels
    print(f"[smoke] black pixels: {black_pixel_count}/{total_pixels} = {black_ratio:.4f}")

    # Expected: 9 blocks * 128 * 128 = 147456 pixels = 0.5625 ratio
    expected_black = 9 * 128 * 128
    expected_ratio = expected_black / total_pixels
    print(f"[smoke] expected: {expected_black} pixels = {expected_ratio:.4f}")

    # Verify block count
    # Find unique block indices that are black
    block_size = 128
    num_blocks_x = 512 // block_size  # 4
    num_blocks_y = 512 // block_size  # 4
    black_blocks = set()
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y1, y2 = by * block_size, (by + 1) * block_size
            x1, x2 = bx * block_size, (bx + 1) * block_size
            block = masked_arr[y1:y2, x1:x2, :]
            if (block == 0).all():
                black_blocks.add((bx, by))
    print(f"[smoke] black blocks: {len(black_blocks)} (expected 9)")
    print(f"[smoke] black block coords: {sorted(black_blocks)}")

    # Verify non-black pixels are unchanged
    non_black_mask = ~black_mask
    original_non_black = arr[non_black_mask]
    masked_non_black = masked_arr[non_black_mask]
    if np.array_equal(original_non_black, masked_non_black):
        print("[smoke] PASS: non-black pixels unchanged")
    else:
        print("[smoke] FAIL: non-black pixels modified!")
        return 1

    # Verify block count
    if len(black_blocks) != 9:
        print(f"[smoke] FAIL: expected 9 black blocks, got {len(black_blocks)}")
        return 1
    print("[smoke] PASS: 9 black blocks (60% of 16)")

    # Verify determinism: same seed -> same mask
    rng2 = random.Random(42)
    masked2 = apply_block_mask(img, mask_ratio=0.6, block_size=128, rng=rng2)
    if np.array_equal(np.array(masked), np.array(masked2)):
        print("[smoke] PASS: determinism (same seed -> same mask)")
    else:
        print("[smoke] FAIL: non-deterministic with same seed")
        return 1

    # Verify different seed -> different mask
    rng3 = random.Random(123)
    masked3 = apply_block_mask(img, mask_ratio=0.6, block_size=128, rng=rng3)
    if not np.array_equal(np.array(masked), np.array(masked3)):
        print("[smoke] PASS: different seed -> different mask")
    else:
        print("[smoke] FAIL: same mask with different seed")
        return 1

    # Verify ratio=0 -> no change
    rng4 = random.Random(42)
    no_mask = apply_block_mask(img, mask_ratio=0.0, block_size=128, rng=rng4)
    if np.array_equal(np.array(img), np.array(no_mask)):
        print("[smoke] PASS: ratio=0 returns original image")
    else:
        print("[smoke] FAIL: ratio=0 modified image")
        return 1

    # Save a visual example
    out_dir = Path(__file__).resolve().parent.parent / "exp" / "smoke_blockmask"
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / "original.png")
    masked.save(out_dir / "masked_r06_b128.png")
    print(f"[smoke] saved visual examples to {out_dir}")

    print("[smoke] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
