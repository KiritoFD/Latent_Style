# attn_sa_mod Fast Curve Read

- Curve CSV: `clip_lpips_curve.csv`
- Formal local fast-eval point:
  - root:
    - [round1_attn_sa_mod_fast_local](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_fast_local)
  - `epoch_0001`
    - full `clip_style / content_lpips = 0.7178 / 0.4651`
    - transfer `clip_style / content_lpips = 0.6955 / 0.4686`
    - wall total `= 58.06s`
  - this is the current authoritative formal fast point for the live lane
  - convergence snapshot:
    - `best_epoch = epoch_0001`
    - `best_in_newest_2 = false`
    - `converged = false`
- Current formal fast screen through settled `epoch_0012`:
  - best pure transfer `CLIP-S`:
    - `epoch_0001`
    - transfer `0.6955 / 0.4686`
  - best transfer `LPIPS`:
    - `epoch_0008`
    - transfer `0.6920 / 0.4416`
  - best all-pairs `CLIP-S`:
    - `epoch_0003`
    - full `0.7180 / 0.4509`
  - current latest settled point:
    - `epoch_0012`
    - transfer `0.6937 / 0.4516`
    - full `0.7169 / 0.4478`
  - local pull is already ahead by one checkpoint:
    - `epoch_0013.pt` is present under the local checkpoint mirror
    - `epoch_0013/summary.json` is not settled yet, so the authoritative curve currently stops at `epoch_0012`
  - convergence snapshot now:
    - `row_count = 12`
    - `best_epoch = epoch_0001`
    - `since_best = 11`
    - `best_in_newest_2 = false`
    - `converged = false`
  - Image-backed localreview status:
    - image-backed reruns already landed for:
      - `epoch_0001`
      - `epoch_0002`
      - `epoch_0003`
  - current localreview shortlist is stale relative to the live fast curve:
    - bestfew handoff still covers `epoch_0002 / epoch_0001 / epoch_0003`
    - it does not yet cover the current best transfer-LPIPS point `epoch_0008`
  - the Windows local GPU lock bug has now been fixed to bind to the child eval pid
  - the pre-fix detached `IntroStyle` process has already been cleared from the machine
  - local deep review is intentionally paused until the bestfew shortlist is rebuilt
- Bootstrap-only historical fast-eval point:
  - `epoch_0001`
  - full:
    - `clip_style = 0.7266`
    - `content_lpips = 0.4322`
  - transfer:
    - `clip_style = 0.7032`
    - `content_lpips = 0.4361`
  - wall total:
    - `119.87s`
- Immediate interpretation:
  - relative to the current internal balanced anchor `LBM-Knee`, the formal line is still in the familiar `style-up / LPIPS-up` regime
  - style remains front-loaded at `epoch_0001`, but the line discovered a materially better LPIPS basin at `epoch_0008`
  - the line is not monotone improving; it is trading style and structure across epochs instead of producing a clean new Pareto frontier
  - keep the lane running until the patience rule becomes meaningful

<!-- ROUND1_AUTO_STATUS:START -->
## Auto Status

- Fast root: [full_eval_fast_local](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_fast_local/full_eval_fast_local)
- Curve CSV: [clip_lpips_curve.csv](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/round1_attn_sa_mod_fast_local/full_eval_fast_local/clip_lpips_curve.csv)
- Best transfer `CLIP-S`:
  - `epoch_0001`
  - `style / lpips = 0.6955 / 0.4686`
- Best transfer `LPIPS`:
  - `epoch_0008`
  - `style / lpips = 0.6920 / 0.4416`
- Best all-pairs `CLIP-S`:
  - `epoch_0003`
  - `style / lpips = 0.7180 / 0.4509`
- Latest settled point:
  - `epoch_0024`
  - transfer `style / lpips = 0.6926 / 0.4508`
  - full `style / lpips = 0.7161 / 0.4470`
  - wall `= 59.65s`
- Convergence snapshot:
  - `row_count = 24`
  - `best_epoch = epoch_0001`
  - `since_last_pareto = 12`
  - `best_in_newest_2 = False`
  - `tail_flat = True`
  - `criterion = joint_transfer_allpairs_pareto`
  - `converged = True`
<!-- ROUND1_AUTO_STATUS:END -->
































