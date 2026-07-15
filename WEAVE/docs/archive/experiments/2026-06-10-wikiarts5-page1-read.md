# WikiArts-5 Page-1 Read

Date: 2026-06-10

Scope:

- answer the current new-dataset question without mixing in unevaluated mainline claims
- reuse the fixed Distinct5 five-style test split as the held-out board
- redraw page-1 style summary assets for the new `wikiarts5` training run

Live note:

- the auto block below is the authoritative current read
- it is refreshed from the live `SaMAM wikiarts5` curve whenever the baseline status updater runs

<!-- WIKIARTS5_PAGE1_AUTO_STATUS:START -->
## Auto Status

- Fixed held-out test split: `G:\GitHub\Latent_Style\Dataset\distinct5_512\test`
- Summary JSON: [wikiarts5_page1_summary.json](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/wikiarts5_page1/wikiarts5_page1_summary.json)
- Summary CSV: [wikiarts5_page1_summary.csv](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/wikiarts5_page1/wikiarts5_page1_summary.csv)
- Curve CSV: [wikiarts5_page1_curve.csv](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/wikiarts5_page1/wikiarts5_page1_curve.csv)
- Summary figure: [fig_wikiarts5_page1_summary.png](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_wikiarts5_page1_summary.png)
- Qualitative figure: [fig_wikiarts5_qualitative_main.png](G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_wikiarts5_qualitative_main.png)
- `IDT` floor:
  - `transfer CLIP-S = 0.6399`
  - `all-pairs CLIP-S = 0.6801`
- Old `SaMAM-2250`:
  - `transfer CLIP-S / LPIPS = 0.5523 / 0.3605`
  - `delta_idt_transfer = -0.0877`
- New `wikiarts5` best transfer-`CLIP-S`:
  - `step = 5750`
  - `transfer CLIP-S / LPIPS = 0.6173 / 0.3504`
  - `delta_idt_transfer = -0.0226`
- New `wikiarts5` best transfer-`LPIPS`:
  - `step = 19500`
  - `transfer CLIP-S / LPIPS = 0.5999 / 0.2209`
  - `delta_idt_transfer = -0.0401`
- Latest settled checkpoint:
  - `step = 23000`
  - `transfer CLIP-S / LPIPS = 0.5953 / 0.2246`
  - `delta_idt_transfer = -0.0446`
<!-- WIKIARTS5_PAGE1_AUTO_STATUS:END -->






























Current status:

- `IDT CLIP-S`:
  - already measured on the fixed five-style test split
  - transfer-only `CLIP-S = 0.6399`
  - all-pairs `CLIP-S = 0.6801`
  - because `wikiarts5` only changes the train pool and keeps the test split fixed, this remains the correct `IDT` floor for the new-dataset read
- current `SaMAM wikiarts5 patch8 segmented` run:
  - result root:
    - [samam_wikiarts5_patch8_segmented_20260610_094447](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447)
  - authoritative live status:
    - [baseline_live_status.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_20260610_094447/baseline_live_status.json)
  - convergence:
    - still `false` as of `2026-06-10`
- current round-1 mainline on the new latent train root:
  - family:
    - `attn_gated_spade`
  - status:
    - training only
    - no local fast-eval curve yet
  - therefore:
    - there is still no trustworthy `our model on wikiarts5` quantitative read to promote or reject

Current measurable read:

- old `SaMAM-2250` on the fixed test split:
  - transfer `CLIP-S / LPIPS = 0.5523 / 0.3605`
  - relative to `IDT`:
    - `delta_idt_transfer = -0.0877`
- new `SaMAM wikiarts5` best transfer-`CLIP-S` checkpoint so far:
  - `step = 5750`
  - transfer `CLIP-S / LPIPS = 0.6173 / 0.3504`
  - relative to `IDT`:
    - `delta_idt_transfer = -0.0226`
  - read:
    - the larger train pool closes most of the old `IDT` gap, but still does not clear the floor
- new `SaMAM wikiarts5` best transfer-`LPIPS` checkpoint so far:
  - `step = 15500`
  - transfer `CLIP-S / LPIPS = 0.6014 / 0.2365`
  - relative to `IDT`:
    - `delta_idt_transfer = -0.0385`
  - read:
    - structure/artifact quality improves substantially more than the old run, but the style move remains below `IDT`
- latest settled checkpoint:
  - see the auto block above for the current settled step
  - relative to `IDT`:
    - currently still negative

Decision read:

- the new `wikiarts5` train pool clearly helps `SaMAM`
- the help is real on both axes:
  - style move improves a lot versus old `SaMAM-2250`
  - LPIPS drops materially at later checkpoints
- but the current run has still not crossed the `IDT` floor on transfer-only `CLIP-S`
- so the paper-safe interim statement is:
  - `wikiarts5` makes the baseline much stronger, but it still has not yet turned `SaMAM` into an above-`IDT` method on the fixed five-style board

Page-1 assets:

- summary table:
  - [wikiarts5_page1_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/wikiarts5_page1/wikiarts5_page1_summary.csv)
  - [wikiarts5_page1_summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/wikiarts5_page1/wikiarts5_page1_summary.json)
- curve table:
  - [wikiarts5_page1_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/wikiarts5_page1/wikiarts5_page1_curve.csv)
- page-1 summary figure:
  - [fig_wikiarts5_page1_summary.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_wikiarts5_page1_summary.png)
  - [fig_wikiarts5_page1_summary.pdf](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_wikiarts5_page1_summary.pdf)
- qualitative companion:
  - [fig_wikiarts5_qualitative_main.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_wikiarts5_qualitative_main.png)
  - [fig_wikiarts5_qualitative_main.pdf](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_wikiarts5_qualitative_main.pdf)

Implementation:

- generator:
  - [scripts_gen_wikiarts5_page1_assets.py](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/scripts_gen_wikiarts5_page1_assets.py)
- it scans every retained `eval_step_*` checkpoint under the live `wikiarts5` `SaMAM` run
- it computes:
  - all-pairs
  - transfer-only
  - identity-only
- it then writes both the machine-readable summary and the figures
