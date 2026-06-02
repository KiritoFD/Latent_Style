# Repo Cleanup And Archive Pass

Date: 2026-06-03

## What changed

This pass reduces the active working surface for the AAAI 2027 push without
throwing away provenance.

### Config cleanup

- Added `configs/README.md` to separate current paper-facing configs from
  archived diagnostic/provenance configs.
- Archived local WSL WikiArt512 probe configs under:
  - `configs/archive/20260603_local_wsl_wikiart512/`
- Archived legacy refactor baseline configs under:
  - `configs/archive/20260603_refactor_legacy/`

### Archive indexing

- Added `archives/README.md` so historical bundles are discoverable instead of
  behaving like an opaque file dump.

## Why this pass matters for writing

The current paper needs a cleaner evidence graph:

- current configs should map to current experiments;
- retired local/timing/refactor configs should remain auditable but not clutter
  the active surface;
- future reviewer-driven experiment blocks need one obvious place to start.

## Keep / archive boundary

Keep in top-level `configs/`:

- current Distinct5-512 family
- current tokenizer exploration family
- reusable base/calibration configs
- `exp_sanity.json`

Archive out of top-level `configs/`:

- local WSL continuation / timing probes
- legacy refactor configs tied to older baseline-preservation work

## Follow-up still needed

1. Normalize paper figure asset paths so active figures live in one obvious
   place.
2. Reduce comparison-table provenance heterogeneity before the next paper
   wording escalation.
3. Run the next paper-closing experiment block with one log-first artifact
   bundle under `docs/experiments/`.
