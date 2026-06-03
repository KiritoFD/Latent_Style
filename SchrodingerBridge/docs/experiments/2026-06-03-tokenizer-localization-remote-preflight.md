Remote tokenizer-localization preflight (superseded legacy chain)

Date: 2026-06-03
Owner surface: remote RTX 3060 / experiment-note only

Protocol checked at the time:
- `G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-localization-probe-protocol.md`

Status note:

- this preflight validated the older legacy256 tokenizer-`t01` resume chain only;
- after review, that chain was rejected as the active packet because it does not
  match the current Distinct5-512 paper-facing mechanism surface;
- keep this note only as path-truth for the rejected legacy route.

Remote code root checked:
- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`

Required resume chain

1. Backbone checkpoint for Arm A
- status: found
- exact remote path:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\tokenizer_t01_factorized_backbone_e16\epoch_0016.pt`
- old-workspace fallback check:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\tokenizer_t01_factorized_backbone_e16\epoch_0016.pt`
  - status: missing

2. Warmup checkpoint for Arm B
- status: found
- exact remote path:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\tokenizer_t01_direct_atom_residual_warmup_e2_from_backbone_e16\epoch_0002.pt`
- old-workspace fallback check:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\tokenizer_t01_direct_atom_residual_warmup_e2_from_backbone_e16\epoch_0002.pt`
  - status: missing

Config launchability on remote without silent path surgery

Arm A config
- config path:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\configs\tokenizer_t01_direct_atom_residual_tokonly_from_backbone_e16.json`
- config file on remote: found
- declared resume checkpoint:
  - `exp/tokenizer_t01_factorized_backbone_e16/epoch_0016.pt`
- declared output dir:
  - `./exp/tokenizer_t01_direct_atom_residual_warmup_e2_from_backbone_e16`
- inherited data root:
  - `../../Latent_Style/latent-256`
- remote resolve result from clean worktree cwd:
  - resolves to `I:\Github\Latent_Style\latent-256`
  - status: found
- inherited eval/test paths:
  - `../../Latent_Style/style_data/overfit50`
  - `../../Latent_Style/eval_cache`
  - `../../Latent_Style/eval_cache/hf`
  - status: all found

Arm B config
- config path:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\configs\tokenizer_t01_direct_atom_residual_frozen_tok_fresh_lancet_e16.json`
- config file on remote: found
- declared resume checkpoint:
  - `exp/tokenizer_t01_direct_atom_residual_warmup_e2_from_backbone_e16/epoch_0002.pt`
- declared output dir:
  - `./exp/tokenizer_t01_direct_atom_residual_frozen_tok_fresh_lancet_e16`
- inherited data root:
  - via base chain, `../../Latent_Style/latent-256`
- remote resolve result from clean worktree cwd:
  - resolves to `I:\Github\Latent_Style\latent-256`
  - status: found
- inherited eval/test paths:
  - `../../Latent_Style/style_data/overfit50`
  - `../../Latent_Style/eval_cache`
  - `../../Latent_Style/eval_cache/hf`
  - status: all found

Path-truth conclusion for the rejected legacy route

- The required resume chain for the protocol is available on the remote clean worktree.
- The two named configs are launchable on remote without additional path surgery, provided they are run from:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`
- Important path truth:
  - both resume checkpoints are present only under the clean worktree `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\...`
  - the inherited dataset/eval relative paths resolve into the older sibling root `I:\Github\Latent_Style\...`, and those resolved targets exist

Supersession conclusion

- technically launchable for the old legacy256 route;
- inadmissible as the current paper-facing tokenizer localization packet;
- superseded by the Distinct5 `L e1` localization packet and its new preflight.
