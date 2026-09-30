# Distinct5 Path-Stability Probe Protocol

Updated: 2026-06-03

Status: prepared, not yet launched in this session

## Why this exists

The paper currently makes a bounded kinetic/path-energy claim:

- kinetic regularization is a practical path stabilizer;
- it is not presented as a globally solved transport theorem;
- the missing empirical support on Distinct5 is a matched `H`-family velocity-field probe against weakened-kinetic controls.

This packet is the missing bridge between that text claim and a paper-safe
measured artifact.

## What is matched

- dataset: `Distinct5-512`
- source: ordinary WikiArt style classes
  - `Early_Renaissance`
  - `Impressionism`
  - `Minimalism`
  - `Rococo`
  - `Ukiyo_e`
- split policy: same `1000 / class` train and `30 / class` test packet already
  used for Distinct5 formal evaluation
- base family: current `H` queue family
- seed: `42`
- formal training target: remote `RTX 3060`, batch `44`

This is not a hand-built adversarial split. The only selection rule is style
separation within standard WikiArt classes. If a matched kinetic ablation fails
here, the burden is on the transport design or the success criterion, not on
the dataset for being exotic.

## Config packet

Prepared configs:

- `configs/aaai2027/path_kinetic_h_base_seed42_b44_base.json`
- `configs/aaai2027/path_kinetic_h_base_seed42_b44_k025.json`
- `configs/aaai2027/path_kinetic_h_base_seed42_b44_k000.json`

Intended interpretation:

- `base`: current reviewed `H` family reference
- `k025`: weakened kinetic control
- `k000`: no-kinetic destructive control

## Probe tool

Prepared tool:

- `tools/probe_path_stability.py`

Outputs:

- `summary.json`
- `per_time_stats.csv`
- `run_summary.csv`
- `fig_velocity_over_time.pdf`

## Measurement contract

The probe is explicitly designed to stay faithful to the executed objective:

- if a checkpoint resolves to `objective_mode = omf`, the default probe mode is
  `field`, not counterfactual multi-step rollout;
- the tool samples `v_\theta(x, t, s)` over time on the same content latents and
  target-style ids used by the packet;
- identity and transfer directions are separated;
- endpoint displacement is still measured from the actual endpoint rule used by
  the checkpoint.

This keeps the readout honest for the current Distinct5 family, where the
active path is better described as a learned endpoint-delta field with kinetic
control than as a long diffusion-style sampler.

## Planned remote command shape

Training:

```bash
py -3 src/run.py --config configs/aaai2027/path_kinetic_h_base_seed42_b44_k025.json
py -3 src/run.py --config configs/aaai2027/path_kinetic_h_base_seed42_b44_k000.json
```

Probe:

```bash
py -3 tools/probe_path_stability.py ^
  --latent-root /mnt/i/wikiart_distinct5_samam_512_latents_ema/train ^
  --classes Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e ^
  --run H_main=/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_h_hard_explore_queue_e3/.../epoch_000X.pt ^
  --run H_k025=/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_path_kinetic_h_base_seed42_b44_k025/epoch_000X.pt ^
  --run H_k000=/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_path_kinetic_h_base_seed42_b44_k000/epoch_000X.pt ^
  --output-dir /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_path_stability_probe
```

## Accept / reject rule

This packet is worth promoting into the paper only if at least one of the
following is cleanly visible under matched evaluation:

1. weakening or removing kinetic clearly raises velocity magnitude or
   path-length ratio in transfer directions; or
2. the full model retains lower path-energy statistics at similar or better
   endpoint movement.

If neither pattern lands cleanly, the paper should keep the kinetic claim at
the current bounded historical-ablation level rather than promoting it into a
Distinct5 mechanism closure.
