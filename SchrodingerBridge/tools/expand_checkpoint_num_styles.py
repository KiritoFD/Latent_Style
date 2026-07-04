"""Expand a checkpoint's num_styles-dependent parameters from old_num_styles to new_num_styles.

Parameters with shape[0] == old_num_styles are expanded:
  - First `old_num_styles` rows are copied from the source checkpoint.
  - Remaining rows are initialized by repeating the last source row with small noise.

Usage:
  python tools/expand_checkpoint_num_styles.py \
    --src exp/FCSB/local_t/630_local_t11_stochastic_dwt_p08/epoch_0005.pt \
    --dst exp/fewshot8_new3/t11_expanded_num_styles8.pt \
    --old-num-styles 5 --new-num-styles 8
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def expand_param(
    tensor: torch.Tensor,
    old_n: int,
    new_n: int,
    *,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """Expand first dim from old_n to new_n, copying first old_n rows."""
    if tensor.ndim == 0 or tensor.shape[0] != old_n:
        return tensor
    new_tensor = torch.empty((new_n, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    new_tensor[:old_n] = tensor
    # Initialize new rows by repeating the last old row + small noise
    last_row = tensor[-1:].expand(new_n - old_n, *tensor.shape[1:])
    noise = torch.randn_like(last_row) * noise_std
    new_tensor[old_n:] = last_row + noise
    return new_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source checkpoint path")
    parser.add_argument("--dst", required=True, help="Destination checkpoint path")
    parser.add_argument("--old-num-styles", type=int, required=True)
    parser.add_argument("--new-num-styles", type=int, required=True)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src_path = Path(args.src).resolve()
    dst_path = Path(args.dst).resolve()
    old_n = int(args.old_num_styles)
    new_n = int(args.new_num_styles)
    if new_n <= old_n:
        raise ValueError(f"new_num_styles ({new_n}) must be > old_num_styles ({old_n})")

    print(f"Loading: {src_path}")
    state = torch.load(src_path, map_location="cpu", weights_only=False)

    model_state = state.get("model_state_dict", state)
    expanded_count = 0
    skipped_count = 0
    for key, tensor in list(model_state.items()):
        if not torch.is_tensor(tensor):
            continue
        if tensor.ndim > 0 and tensor.shape[0] == old_n:
            old_shape = tuple(tensor.shape)
            new_tensor = expand_param(tensor, old_n, new_n, noise_std=args.noise_std)
            new_shape = tuple(new_tensor.shape)
            model_state[key] = new_tensor
            print(f"  EXPAND  {key}: {old_shape} -> {new_shape}")
            expanded_count += 1
        else:
            skipped_count += 1

    state["model_state_dict"] = model_state
    print(f"\nSummary: expanded={expanded_count} skipped={skipped_count}")

    if args.dry_run:
        print("[DRY RUN] Not saving.")
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, dst_path)
    print(f"Saved: {dst_path}")


if __name__ == "__main__":
    main()
