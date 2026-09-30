# Tokenizer Probe Checkpoint Reselection Policy

Date: 2026-06-03

Scope: paper-safe reselection policy for the tokenizer execution-alignment
packet if the selected `H e1` checkpoint remains unavailable on remote.

## Policy goal

The execution-alignment probe is meant to stay attached to the **reviewed H
mainline mechanism family**. The reselection rule must therefore preserve claim
continuity before it preserves convenience.

## 1. Default target

Primary target remains:

- family: `H`
- checkpoint: `epoch_0001`

Reason:

- the packet README identifies `H e1` as the intended lower-LPIPS balance point
  inside the currently reviewed H family
- the master experiment log still treats H-family points as the reviewed
  mainline reference family for this theory lane

## 2. When `H e2` is acceptable

`H e2` is acceptable **only** as a same-family adjacent-epoch contingency if
all of the following are true:

1. `H e1` remains missing after explicit recovery search
2. `H e2` is available and belongs to the exact same remote family / config
3. the probe is still described as an `H`-family execution-alignment packet,
   not as a newly chosen mainline
4. the memo / log explicitly records that the selection changed because the
   intended `H e1` payload was unavailable, not because `H e2` is theoretically
   preferred

Paper-safe interpretation:

- `H e2` preserves mechanism continuity because it stays inside the same
  tokenizer / routing / queue family
- it weakens the original "lower-LPIPS balance point" rationale, but does not
  change the family-level theory object

## 3. When switching to `F` / `J` / `K`-like points is too large a jump

Switching away from `H` to `F`, `J`, `K`, or similar points is too large a
mechanism jump for this packet if the goal is to preserve the same tokenizer
execution-alignment claim.

Reason:

- `F` changes the queue curriculum story
- `J` changes the auxiliary target / hard-SWD story
- `K` changes the content-adaptive VQ routing story

These are not checkpoint substitutions inside one mechanism family. They are
family changes that alter the object being probed.

Paper-safe rule:

- do **not** silently replace `H e1` with `F/J/K`
- if a family switch becomes necessary, treat it as a **new packet** with a new
  claim boundary, not as checkpoint reselection within the old packet

## 4. Reselection metadata that must be preserved

If `H e2` is used, preserve all of the following in the landed packet:

1. original intended target:
   - `H e1`
2. proof-of-absence summary:
   - searched roots
   - what artifacts survived
   - that no silent substitute was used first
3. exact replacement target:
   - family
   - epoch
   - checkpoint path
   - resolved config path
4. continuity note:
   - "same-family adjacent-epoch reselection due to unavailable `H e1` payload"
5. evaluation metadata:
   - full and transfer metrics used to characterize the replacement point
   - hardware
   - branch / commit or synced workspace identifier
6. script-level label continuity:
   - if downstream tables or scripts still include both `H e1` and `H e2`
     labels, record which one is actually used by the probe packet

## 5. Safe wording boundary after reselection

If `H e2` is used:

- safe:
  - "the execution-alignment probe remained in the reviewed H family after the
    originally selected `H e1` payload could not be recovered"
  - "the packet was reselected to a same-family adjacent epoch for continuity"

- unsafe:
  - "`H e2` is theoretically equivalent to `H e1` in every respect"
  - "the reselection is irrelevant"
  - any wording that implies the probe still targets the original lower-LPIPS
    balance point without noting the change

If a switch to `F/J/K` occurs:

- safe only if the packet is renamed and re-scoped as a new mechanism-family
  probe
- unsafe if presented as a mere checkpoint fallback

## 6. One-line operational rule

If `H e1` remains unavailable, reselect to `H e2` only as an explicitly
documented same-family adjacent-epoch fallback; do not jump to `F/J/K`-style
points unless the packet is re-declared as a new mechanism-family probe.
