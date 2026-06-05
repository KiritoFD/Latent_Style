# Local `eval_cache` Manual Retention Policy - 2026-06-05

Scope:

- Local root: `G:\GitHub\Latent_Style\eval_cache`
- Purpose: distinguish real experiment checkpoints from evaluation/model caches before cleanup.
- Rule used here: a `.pt`, `.pth`, `.bin`, `.onnx`, or `.data` file is not treated as deletable until its directory role and references have been checked.

This pass is intentionally manual. Directory-size scans were used only to find candidates; each class below was opened at file level, and representative configs/logs/source references were checked before assigning a cleanup decision.

## Summary

`eval_cache` is an evaluation and dependency cache surface, not a training checkpoint dump. It contains:

- ArtFID metric dependency: `artfid/art_inception.pth`.
- HF/ModelScope model caches: CLIP and VAE snapshots.
- A complete manual CLIP cache: `manual_clip/openai-clip-vit-base-patch32`.
- DINO/offline-pairing cache: 9636-row DINOv2 embedding cache and top-k pairing tables.
- Full-eval reference feature caches: `ref_feats_*.pt`.
- VAE compile and ONNX speed-path artifacts.

Cleanup decision:

- Keep all valid model/eval/cache artifacts for now.
- Delete only the invalid HF `.incomplete` blob and two empty ModelScope temp directories after recording them in `cleanup/manual_cache_cleanup_20260605.csv`.
- Do not delete `offline_pairing`, `manual_clip`, `artfid`, valid VAE blobs, `ref_feats_*.pt`, `vae_compile`, or `vae_onnx` in this pass.

## File Classes Checked

| Class | Size MB | Manual evidence opened | Decision |
|---|---:|---|---|
| `artfid` | 102.936 | `art_inception.pth`; ArtFID loader/log references | Keep |
| `hf` | 1332.631 | `refs/main`, VAE configs, ModelScope cache files | Partial cleanup only |
| `manual_clip` | 580.658 | CLIP `config.json`, tokenizer files, `pytorch_model.bin` | Keep |
| `offline_pairing` | 3632.707 | DINO cache logs, top4/top8 files, source references | Keep |
| `ref_feats_*.pt` | 9.426 | `run_evaluation.py` cache logic; every file read-only loaded | Keep |
| `vae_compile` | 50.541 | Triton/Inductor `.ptx/.llir/.cubin` artifacts | Keep for now |
| `vae_onnx` | 95.035 | `decoder.onnx`, `decoder.onnx.data`, export docs | Keep for now |

## Detailed Findings

### `offline_pairing`

Opened files:

- `dinov2_cache_stdout.log`
- `dinov2_cache_stderr.log`
- `dinov2_small_train_cache.pt`
- `dinov2_small_train_pairing_top4.pt`
- `dinov2_small_train_pairing_top8.pt`

The stdout log records:

- `image_root = G:\GitHub\Latent_Style\style_data\train`
- `latent_root = G:\GitHub\Latent_Style\latent-256`
- styles: `photo`, `Hayao`, `monet`, `vangogh`, `cezanne`
- `n_rows = 9636`
- model: `facebook/dinov2-small`

Source references were found in:

- `SchrodingerBridge/tools/experiments/run_orthogonal_budget36.py`
- `SchrodingerBridge/tools/experiments/build_offline_dino_pairing_cache.py`
- `SchrodingerBridge/tools/experiments/build_offline_dino_pairing_plan.py`
- `SchrodingerBridge/src/utils/dataset.py`

Decision: keep. This is a large cache, but it is attached to the offline DINO pairing experiment family. Deleting it would save about 3.6GB but would force regeneration of the 9636-row DINO embedding cache and pairing tables.

### `hf` and `manual_clip`

Opened evidence:

- `models--openai--clip-vit-base-patch32/refs/main`
- `models--stabilityai--sd-vae-ft-mse/refs/main`
- `models--madebyollin--sdxl-vae-fp16-fix/refs/main`
- `models--stabilityai--sd-vae-ft-mse/.../config.json`
- `models--madebyollin--sdxl-vae-fp16-fix/.../config.json`
- `manual_clip/openai-clip-vit-base-patch32/config.json`

The valid cache contents are model dependencies:

- `openai/clip-vit-base-patch32` for CLIP evaluation.
- `stabilityai/sd-vae-ft-mse` VAE.
- `madebyollin/sdxl-vae-fp16-fix` VAE.
- ModelScope copy of `stabilityai/sd-vae-ft-ema`.

One invalid file was found:

- `eval_cache/hf/models--openai--clip-vit-base-patch32/blobs/a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f.incomplete` at 55.994MB.

The manual CLIP directory contains a complete `pytorch_model.bin` plus tokenizer/config files, so the incomplete HF blob is not needed as evidence. It will be deleted after being recorded in the cleanup CSV.

### `ref_feats_*.pt`

Opened evidence:

- Cache logic in `SchrodingerBridge/src/utils/run_evaluation.py` around `cache_file = cache_dir / f"ref_feats_{dataset_hash}_m{max_ref_cache_tag}.pt"`.
- Each local `ref_feats_*.pt` was loaded read-only with `torch.load(..., map_location='cpu')`.

Observed structure:

- dictionary keyed by style id such as `0`, `1`, `2`, `3`, `4`;
- each style id maps to feature lists with lengths matching the `m` cache tag or available references.

Decision: keep. These are small full-eval reference feature caches, not checkpoints. Deleting them saves only 9.426MB and costs eval rebuild time.

### `vae_compile`

Opened evidence:

- `ema_b2_64_reduce_overhead`
- `ema_b4_64_reduce_overhead`
- Triton/Inductor generated `.ptx`, `.llir`, and `.cubin` files.
- `fast_infer_ablate43/README.md` documents a torch-compile speed path.

Decision: keep for now. This is probably regenerable, but deleting it would erase speed-path context while timing archaeology is still active.

### `vae_onnx`

Opened evidence:

- `ema_b2_64/decoder.onnx`
- `ema_b2_64/decoder.onnx.data`
- `ema_b2_64/decoder.json`
- `fast_infer_ablate43/export_onnx.py`

Decision: keep for now. It is an ONNX export artifact, not a ckpt. It can become a cleanup candidate only if the ONNX/TensorRT path is abandoned or regenerated elsewhere.

## Cleanup Boundary

Safe cleanup in this pass:

- invalid `.incomplete` HF blob;
- empty `._____temp` ModelScope directories.

Not cleaned:

- any valid VAE/CLIP/ArtFID dependency;
- DINO/offline-pairing cache;
- full-eval reference feature caches;
- VAE compile and ONNX speed-path artifacts.

## Follow-Up

The next disk-recovery target should not be root `eval_cache`. Better candidates require separate manual policies:

- remote `SchrodingerBridge/exp` epoch thinning;
- remote SaMAM central checkpoint thinning;
- local and remote dataset/cache/archive policies.
