# WEAVE Documentation

**Last updated:** 2026-07-15

## Start Here

| Path | Role |
|---|---|
| `713/SUBMISSION_HANDOFF_2026-07-15.md` | Current repository, remote execution, method, reproduction, and next-experiment handoff. |
| `reproduction/baseline_reproduction.md` | Clean 15-epoch from-scratch baseline and per-epoch selection evidence. |
| `reproduction/root_layout_equivalence.md` | Exact root-layout model equivalence and full-board evaluation check. |
| `reproduction/endpoint_adain_axis.csv` | Fully evaluated inference-scale axis. |
| `reproduction/hf_oriented_nohh_result.md` | Latest from-scratch architecture experiment. |
| `713/GRADIENT_INFORMATION_FLOW_DEBUG_2026-07-14.md` | Gradient and information-flow diagnosis. |
| `713/HF_ARCHITECTURE_PROBE_2026-07-13.md` | Detailed historical HF-route probe evidence. |
| `archives/README.md` | Repository archive policy and provenance map. |

## Active Contract

- Project name and local directory: `WEAVE`.
- Branch: `submission`.
- Active implementation: project-root Python modules and `utils/`.
- Canonical training config: root `config.json`.
- Canonical inference config: root `inference.json`.
- Local data: `G:\GitHub\Latent_Style\WEAVE\data\train` and `data\test`.
- Remote data: `I:\Github\Latent_Style\WEAVE\data\train` and `data\test`.
- Tracked configs use only project-relative paths.
- Submission training starts from a fresh initialization and uses no frozen adapter or image/latent post-processing.

## Paper Bundle

The active manuscript remains under `aaai2027_v4/`. Historical paper
workspaces, machine-specific tool instructions, launchers, configs, and the old
`src/` implementation are retained under `archives/` only for provenance.

## Evidence Rule

DINO-S is the primary style metric and CLIP-S is secondary. DINO-C and LPIPS
must be reported to reject content-collapse gains. Select checkpoints directly
from per-epoch metrics; do not introduce a custom mixed score.

Historical documents may contain the former `SchrodingerBridge` directory name.
Those strings identify where an artifact was originally produced and are not
current execution instructions.
