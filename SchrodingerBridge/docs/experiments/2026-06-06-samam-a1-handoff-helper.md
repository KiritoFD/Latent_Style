# SaMam To A1 Handoff Helper

Date: 2026-06-06

Purpose:

- reduce manual polling between the bounded `latent SaMam` side quest and the
  main `A1` Distinct5-512 packet
- keep the remote `3060` on a strict single-lane policy under `< 11.0 GiB`

Tool:

- [handoff_remote_latent_samam_to_a1.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/handoff_remote_latent_samam_to_a1.py)

What it does:

1. checks whether the remote latent `SaMam` run has produced any retained
   checkpoint besides `last.ckpt`
2. checks whether the latent `SaMam` process is still alive
3. checks whether the `A1` remote log already exists
4. if and only if a retained checkpoint exists, it can:
   - stop the latent side quest
   - wait briefly for the lane to clear
   - launch `A1` through the reviewed remote launcher

Important boundary:

- this helper does **not** launch `A1` while the latent lane is still active
  unless explicitly invoked with:
  - `--stop-latent-on-retained`
- if no retained checkpoint exists yet, it exits without side effects

Dry-run example:

```bash
python SchrodingerBridge/tools/experiments/handoff_remote_latent_samam_to_a1.py --dry-run
```

Active handoff example once the first retained checkpoint exists:

```bash
python SchrodingerBridge/tools/experiments/handoff_remote_latent_samam_to_a1.py \
  --stop-latent-on-retained
```
