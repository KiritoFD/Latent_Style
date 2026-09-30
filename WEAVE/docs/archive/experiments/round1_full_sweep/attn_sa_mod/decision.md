# attn_sa_mod Decision

- Status: reject
- Promotion state: rejected
- Finality: final for round-1 promotion purposes
- Frozen evidence packet:
  - [vlm_stageclose_snapshot.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/attn_sa_mod/vlm_stageclose_snapshot.md)

Current evidence stack:

- fast curve:
  - converged through `epoch_0024`
  - best transfer `CLIP-S` remains `epoch_0001`
  - best balanced internal checkpoint remains `epoch_0008`
- `IntroStyle`:
  - all shortlisted checkpoints still show negative transfer style margins
  - `epoch_0008` is the least-bad local structure/style tradeoff, not a positive style-control win
- `DINO`:
  - `epoch_0008` is best on the shortlisted set, but the gain is incremental rather than branch-changing
- frozen external-baseline `VLM`:
  - frozen stageclose board:
    - `e08`
      - `AttnSA_e08 = 2 / 200`
      - `SaMAM_2250 = 104 / 200`
      - `Seedream_repaired750 = 94 / 200`
    - `e24`
      - `AttnSA_e24 = 0 / 169`
      - `SaMAM_2250 = 97 / 169`
      - `Seedream_repaired750 = 72 / 169`

Current read:

- `attn_sa_mod` does not currently solve the real round-1 target.
- It can train stably and produce a clean internal fast-curve closure.
- It does not convert that into target-style competitiveness against the external visual anchors.
- The partial frozen `VLM` board is already strongly negative:
  - `AttnSA_e08` barely wins at all
  - `AttnSA_e24` currently wins nothing
  - both lose heavily to `SaMAM` and `Seedream`

Why this is already enough:

- `e08` has effectively reached full-board scale and remains overwhelmingly negative.
- `e24` is also already large enough to be decisive and still has zero wins.
- `IntroStyle` margins are negative on every shortlisted point.
- `DINO` shows only a modest structure improvement, not a compensating visual win.
- There is no remaining evidence path that is likely to overturn the promotion decision without opening a meaningfully different mechanism family.

Decision rule from the current evidence:

- do not promote `attn_sa_mod` as the next internal frontier
- close the family as a reject for round 1
- do not spend a fresh remote lane on a continuation of this family
- keep it as negative evidence for:
  - stable optimization alone is insufficient
  - self-attention modulation without stronger style-driving control still leaves the branch below `SaMAM / Seedream` on the real board

Remaining follow-up:

- allow the background frozen `VLM` snapshots to keep accumulating as confirmatory evidence
- do not let those remaining rows block the next family queue
