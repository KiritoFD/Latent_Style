# SA-SWD Axis Ablation Launch Manifest

Date: 2026-06-03

## Formal remote packet

This packet should run only on the remote `RTX 3060`.

## Configs

Base:

- `configs/aaai2027/saswd_axis_h_base_seed42_b44.json`

Matched arms:

- `configs/aaai2027/saswd_axis_h_base_seed42_b44_saswd_semantic.json`
- `configs/aaai2027/saswd_axis_h_base_seed42_b44_saswd_random.json`

## Intended launch order

1. semantic arm
2. random arm

## Intended task names

- semantic:
  - `SB_SASWD_H_SEM_S42`
- random:
  - `SB_SASWD_H_RAND_S42`

## Expected output roots

- semantic:
  - `exp/saswd_axis_h_base_seed42_b44_saswd_semantic`
- random:
  - `exp/saswd_axis_h_base_seed42_b44_saswd_random`

## Matched run contract

- seed:
  - `42`
- batch:
  - `44`
- epochs:
  - `3`
- queue family:
  - reviewed `H` queue family
- eval:
  - strict `5x5 / 750` full eval with the standard Distinct5 contract

## Variable under test

Only this field should differ across the two arms:

- `bridge.terminal_swd_axis_source`

Values:

- `semantic`
- `random`
