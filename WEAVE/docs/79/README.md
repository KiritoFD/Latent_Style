# docs/79: dataset and result registry

This directory records the current dataset/protocol map before further main-table cleanup. The immediate goal is to stop mixing three surfaces that have similar names in older notes: D5 / Distinct5, P2A-256, and R5 / Random5 / random20.

## 0. Files in this registry

| File | Purpose |
|---|---|
| `README.md` | Dataset definitions, style sets, naming policy, and current local result inventory summary. |
| `results_manifest.csv` | Machine-readable local `results/` image-packet registry by dataset and method. |
| `main_table_v4_remote.csv` | Local snapshot of remote `I:\results\tables\main_table.csv`, matching the v4 paper main table values. |
| `baseline_result_sources.md` | Evidence map linking v4 table values, local image packets, and remote sources/gaps. |
| `main_table_image_audit.csv` | Row-level join between v4 main-table values and local image-packet evidence. |
| `r5_consolidation_log.md` | Log of R5 packets copied into `results\R5-WikiArt` and remote search status. |

## 1. Canonical dataset definitions

| Short name | Canonical meaning | Resolution | Style/domain set | Standard eval size | Current role |
|---|---|---:|---|---:|---|
| D5 / Distinct5 | Distinct5-WikiArt stress split | 512 | `Early_Renaissance`, `Impressionism`, `Minimalism`, `Rococo`, `Ukiyo_e` | 5 x 5 x 30 = 750 images | Main IDT-calibrated stress benchmark |
| P2A-256 / P256 | Photo2Art-256 CycleGAN-style protocol | 256 | `cezanne`, `Hayao`, `monet`, `photo`, `vangogh` | 5 x 5 x 30 = 750 images | Historical 256px support / external-baseline surface |
| R5-512 / Random5 | Random5 WikiArt hold-out subset | 512 | `Cubism`, `Expressionism`, `Pop_Art`, `Romanticism`, `Symbolism` | 5 x 5 x 30 = 750 images | Generalization subset carved out of the broader random20/wikiarts20 surface |
| random20 / wikiarts20 | 20-family WikiArt training/eval surface | 512 | 20 WikiArt families; exact full list must be read from the run config/manifest before table use | 20 x 20 x 30 = 12000 images | Broader training/generalization surface; sometimes mislabeled as R5 in old notes |

## 2. What each dataset is for

### D5 / Distinct5

D5 is the paper's main stress test. The five styles are intentionally far apart under the IDT/CLIP screening logic: `Early_Renaissance`, `Impressionism`, `Minimalism`, `Rococo`, and `Ukiyo_e`. The important evaluation property is that the unchanged source image is a strong no-op baseline, so raw CLIP-S is not enough; every method should be interpreted relative to the IDT floor.

Known source/evidence paths:

| Kind | Path |
|---|---|
| Config evidence | `G:\GitHub\Latent_Style\SchrodingerBridge\configs\620_spatial_bridge_base.json` |
| Local collected results | `G:\GitHub\Latent_Style\SchrodingerBridge\results\D5-512` |
| Remote source dataset | `I:\datasets\wikiart_distinct5_512_images` |
| Remote baseline map evidence | `G:\GitHub\Latent_Style\SchrodingerBridge\results\_remote_map.json` |

### P2A-256 / P256

P2A-256 is the older 256px Photo2Art protocol. It uses five domains: `cezanne`, `Hayao`, `monet`, `photo`, and `vangogh`. In this protocol, `photo` is a domain in the ordered source-target grid, not a WikiArt style.

Known source/evidence paths:

| Kind | Path |
|---|---|
| Local collected results | `G:\GitHub\Latent_Style\SchrodingerBridge\results\P256` |
| Remote clean baseline root | `I:\exp_256_photo2art` |
| Remote duplicate baseline root | `I:\Github\Latent_Style\exp_baseline_256` |
| Prior inventory note | `G:\GitHub\Latent_Style\SchrodingerBridge\state\remote_inventory.md` |

### R5-512 / Random5

R5 should mean the true random 5-style WikiArt hold-out subset. The current local `results\R5-512` filenames show the five styles as `Cubism`, `Expressionism`, `Pop_Art`, `Romanticism`, and `Symbolism`.

Known source/evidence paths:

| Kind | Path |
|---|---|
| Local random5 baseline subset | `G:\GitHub\Latent_Style\SchrodingerBridge\results\R5-512` |
| Ours random20/R5 eval run | `G:\GitHub\Latent_Style\SchrodingerBridge\exp\630_random20_heun_5ep` |
| Eval summary with random20 source bank | `G:\GitHub\Latent_Style\SchrodingerBridge\exp\630_random20_heun_5ep\eval_r5_baseline\summary.json` |
| Referenced source bank | `G:\GitHub\Latent_Style\Dataset\wikiart_random20_512\wikiart_random20_512\images\test` |

## 3. Naming collision warning

`R5-WikiArt` is not currently safe as a canonical name without inspecting the files. In the local tree, `G:\GitHub\Latent_Style\SchrodingerBridge\results\R5-WikiArt` contains several 750-image method folders whose filenames use the D5 style set (`Early_Renaissance`, `Impressionism`, `Minimalism`, `Rococo`, `Ukiyo_e`). That means the directory name and the style set disagree.

Policy for future cleanup:

| Case | Action |
|---|---|
| Directory contains `Cubism`, `Expressionism`, `Pop_Art`, `Romanticism`, `Symbolism` | Treat as canonical `R5-512` / Random5. |
| Directory contains D5 styles under an `R5-WikiArt` name | Treat as legacy/misplaced D5-style packet until a manifest proves otherwise. |
| Directory points to `baseline_wikiarts20` or `wikiarts20_eval` | Treat as random20/wikiarts20, then derive R5 only with an explicit 5-style subset manifest. |

## 4. Current local result inventory

Image counts below are local `results/` counts as of this pass. `direct` counts files immediately under the method directory; `recursive` includes nested duplicate copies. A clean 750-pair packet should normally have direct=750.

| Dataset dir | Method | Direct images | Recursive images | Status |
|---|---|---:|---:|---|
| `D5-512` | `adain` | 750 | 1500 | usable but has nested duplicate copy |
| `D5-512` | `cut` | 750 | 1500 | usable but has nested duplicate copy |
| `D5-512` | `identity` | 750 | 1500 | usable but has nested duplicate copy |
| `D5-512` | `samam` | 750 | 1500 | usable but has nested duplicate copy |
| `D5-512` | `samst` | 750 | 1500 | usable but has nested duplicate copy |
| `D5-512` | `sdturbo` | 750 | 1500 | usable but has nested duplicate copy |
| `D5-512` | `seedream` | 750 | 791 | usable; extra nested/side files need pruning or manifesting |
| `D5-512` | `stylealigned` | 750 | 750 | clean count |
| `D5-512` | `styleid` | 750 | 750 | clean count |
| `D5-512` | `styleshot` | 745 | 745 | incomplete, missing 5 |
| `D5-512` | `wct` | 750 | 1500 | usable but has nested duplicate copy |
| `D5-512` | `weave` | 750 | 750 | clean count |
| `D5-512` | `zstar` | 750 | 750 | clean count |
| `P256` | `adain` | 750 | 750 | clean count |
| `P256` | `identity` | 750 | 750 | clean count |
| `P256` | `samam` | 750 | 750 | clean count |
| `P256` | `samst` | 750 | 750 | clean count |
| `P256` | `sdturbo` | 750 | 750 | clean count |
| `P256` | `stylealigned` | 750 | 750 | clean count |
| `P256` | `styleid` | 750 | 750 | clean count |
| `P256` | `styleshot` | 750 | 750 | clean count |
| `P256` | `wct` | 750 | 750 | clean count |
| `P256` | `weave` | 750 | 750 | clean count |
| `P256` | `zstar` | 750 | 750 | clean count |
| `P256` | `cut` | 0 | 0 | missing locally in unified results |
| `P256` | `seedream` | 0 | 0 | missing locally in unified results |
| `R5-512` | `stylealigned` | 750 | 750 | clean count |
| `R5-512` | `styleshot` | 740 | 740 | incomplete, missing 10 |
| `R5-512` | `zstar` | 750 | 750 | clean count |
| `R5-WikiArt` | `cut` | 750 | 750 | legacy name; style-set audit needed |
| `R5-WikiArt` | `samam` | 750 | 750 | legacy name; style-set audit needed |
| `R5-WikiArt` | `samst` | 750 | 750 | legacy name; style-set audit needed |
| `R5-WikiArt` | `sdturbo` | 1123 | 1123 | nonstandard count |
| `R5-WikiArt` | `seedream` | 0 | 724 | nested/incomplete; nonstandard count |
| `R5-WikiArt` | `styleid` | 750 | 750 | legacy name; style-set audit needed |

## 5. Next cleanup target

The first machine-readable manifest for every method packet is now:

`G:\GitHub\Latent_Style\SchrodingerBridge\docs\79\results_manifest.csv`

The manifest uses these fields:

| Field | Meaning |
|---|---|
| `dataset_canonical` | One of `D5-512`, `P2A-256`, `R5-512`, `random20`. |
| `dataset_dir` | Existing local directory name. |
| `method` | Baseline or ours method name. |
| `image_root` | Exact local image root used for metrics. |
| `style_set_detected` | Styles parsed from filenames or manifest. |
| `image_count_direct` | Direct image count. |
| `image_count_recursive` | Recursive image count. |
| `status` | `clean_750`, `duplicate_nested`, `incomplete`, `legacy_name_mismatch`, or `missing`. |
| `metric_source` | Summary/CSV/JSON used for table values, if found. |

Files should be moved or renamed inside `results/` only after checking this manifest, because current naming already contains at least one proven collision.

Recommended next actions:

| Priority | Action |
|---:|---|
| 1 | Keep `D5-512` and `P2A-256` as the clean canonical names; optionally add a `P2A-256` alias for the current `P256` directory instead of renaming immediately. |
| 2 | Quarantine or relabel `R5-WikiArt` rows whose detected styles are D5 styles; do not use them as Random5 evidence. |
| 3 | Fill missing/incomplete packets: `P256/cut`, `P256/seedream`, `D5-512/styleshot`, `R5-512/styleshot`. |
| 4 | For random20/wikiarts20, derive a separate manifest from `baseline_wikiarts20` and `wikiarts20_eval` before claiming R5 metrics. |
