# tok_a_dino_dict Plan

- Wave: `wave1_tokenizer`
- Axis: `tokenizer`
- Notes: Universal keys plus style-specific values with DINO-masked SWD.
- Round-1 execution policy:
  - keep backbone frozen
  - train tokenizer path only through `freeze_mode=style_branch`
  - use DINO supervision as a tokenizer-shaping signal, not as a full-backbone joint-training trigger
- Queue policy update:
  - `tok_a_dino_dict` is no longer the next remote family by default
  - DINO-related tokenizer families were pushed to the tail of the remote round-1 order
  - this family should resume only after the non-DINO mainline families and after the DINO cache/tooling path is clean enough
- If direct formal launch remains unstable after DINO cache alignment is fixed:
  - add a tokenizer-only DINO warm-start / pretrain stage
  - then resume the normal round-1 family run from that tokenizer-biased checkpoint
