# Dataset Layout

Training and evaluation use the same project-relative layout on every machine:

```text
data/
  train/
    Early_Renaissance/
    Impressionism/
    Minimalism/
    Rococo/
    Ukiyo_e/
    .latent_cache/
      prototype_pairing_top8.pt
      packed/
  test/
    Early_Renaissance/
    Impressionism/
    Minimalism/
    Rococo/
    Ukiyo_e/
```

The dataset contents are intentionally ignored by Git. On 2026-07-15 the local and
remote copies were moved into this layout rather than referenced through drive-specific
paths. Tracked configuration must use paths relative to the project root.

For the pre-refactor reproduction, `latent_cache_dir` remains
`.latent_cache/packed` to preserve the historical loader behavior exactly. After the
baseline is reproduced, the redundant `packed/packed` lookup will be fixed and checked
for numerical equivalence in a separate commit.
