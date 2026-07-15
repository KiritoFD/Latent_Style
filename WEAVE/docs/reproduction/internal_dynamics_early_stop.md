# Internal-Dynamics Early Stopping

Date: 2026-07-15  
Branch: `submission`

## Question

Can WEAVE select the style checkpoint without generating images or consulting
DINO-S, CLIP-S, LPIPS, or DINO-C during training?

## Diagnostic Result

The ordinary FM residual is not a useful stopping signal. On the original
15-epoch oriented-HF trajectory, residual alignment with the FM target keeps
improving after epoch 4 while retrospective DINO-S declines. Lower training
loss therefore measures endpoint regression, not perceptual style quality.

The useful event is a route-level phase transition:

1. The mean LH/HL target-HF gate changes from contraction to expansion.
2. The shared-trunk LL/HF gradient-norm ratio crosses from above one to below
   one for the first time.
3. The target-HF route gradient becomes much larger relative to the ordinary
   HF velocity head.

On the previously evaluated 15-epoch curve, this event occurs at epoch 4,
which is also the DINO-S maximum (`0.491543`). DINO-S is used only for this
retrospective validation and is absent from the stopping rule.

## Blind Online Run

`experiments/architecture/hf_oriented_internal_early_stop.json` starts from
scratch with the canonical 15-epoch cosine schedule. It uses a fixed batch of
four latents at `t=0.5`, performs two extra graph-preserving backwards per
epoch, and never decodes an image. The monitor selected epoch 4 and terminated
training there:

| epoch | gate delta | shared LL/HF | route/shared HF | route/HF head | event |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000000 | 1.4727 | 0.8323 | 0.1816 | no |
| 2 | -0.000803 | 1.4328 | 0.6133 | 0.1907 | no |
| 3 | -0.000255 | 1.0868 | 0.4984 | 0.2111 | no |
| 4 | +0.003710 | 0.1634 | 0.2370 | 0.4561 | **stop** |

The checkpoint is remote at:

`I:\Github\Latent_Style\WEAVE\runs\submission\hf_oriented_internal_early_stop\epoch_0004.pt`

The full machine-readable curve is in
`docs/reproduction/internal_dynamics_early_stop.csv`.

## Interpretation

Before the transition, the target-HF branch is learning a compact correction:
its gates shrink and LL gradients still dominate the shared representation.
At the transition, HF gradients abruptly dominate the shared trunk while the
target-HF gates begin to grow. Subsequent optimization can continue reducing
the supervised endpoint error, but it increasingly solves that error by
amplifying the HF branch and reorganizing shared features. The earlier
retrospective curve shows that this regime improves FM-target alignment while
reducing DINO-S. The transition is therefore a practical boundary between
learning a useful style direction and over-absorbing the paired endpoint.

## Rule And Scope

The implemented candidate event is:

`gate_delta > 0` and the shared LL/HF gradient ratio crosses `1.0` downward.

The event is measured after the epoch and before checkpoint saving. Probe
forwards run under `eval`, do not call the optimizer, and preserve CPU/CUDA RNG
state with `torch.random.fork_rng`, so the monitor does not perturb the next
training epoch.

This is validated for the oriented-HF architecture and should not yet be
presented as an architecture-independent theorem. For another route design,
the meaningful gate and parameter groups must be defined explicitly, and the
correspondence should be checked retrospectively on at least one complete
curve before enabling automatic stopping.
