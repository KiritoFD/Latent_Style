# AAAI-27 Official Requirement Check

Updated: 2026-06-08

This note is a local compliance memo against the currently visible official AAAI-27 author-facing pages.

## Official pointers

- AAAI-27 conference page / author timetable:
  - https://aaai.org/conference/aaai/aaai-27/
- AAAI supplementary-material policy page:
  - https://aaai.org/conference/aaai/aaai-23/supplementary-material/
- AAAI publication policy / author-kit pointer:
  - https://aaai.org/aaai-publications/aaai-publication-policies-guidelines/

## What the official pages currently say

- `AAAI-27` timetable currently lists:
  - abstract due: `2026-07-21`
  - full paper due: `2026-07-28`
  - supplementary material and code due: `2026-07-31`
- the supplementary-material page allows:
  - technical appendix PDF
  - multimedia appendix ZIP
  - code and data ZIP
- the supplementary-material page also states that:
  - the main submission must remain self-contained
  - reviewers are not obliged to consult supplementary material
  - critical contribution material should stay in the main paper

## Current local submission surface

Main paper:

- source:
  - [paper_aaai2027.tex](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/paper_aaai2027.tex)
- pdf:
  - [paper_aaai2027.pdf](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/paper_aaai2027.pdf)
- local page count:
  - `8` total
  - references begin on page `7`

Supplement:

- source:
  - [supplement_aaai2027.tex](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/supplement_aaai2027.tex)
- pdf:
  - [supplement_aaai2027.pdf](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/supplement_aaai2027.pdf)
- local page count:
  - `4`

## Compliance read

### Good

- a main-paper PDF exists and compiles locally
- a separate supplementary appendix PDF exists and compiles locally
- the main paper now keeps the core claims in the main body:
  - IDT diagnostic
  - selected frontier rows
  - qualitative strip
  - artifact-sensitive table
  - non-CLIP probe summary

### Acceptable but still imperfect

- the supplement is structured like a technical appendix, not a submission ZIP manifest
- code/data are organized locally, but not yet bundled into one explicit anonymized archive
- the current final mirror is assembled in:
  - [aaai2027/final](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/final)

### Remaining risk

- blind preference evidence is still exploratory rather than human or external-VLM verified
- the current ZIP bundle is local and convenience-oriented, not a final anonymized upload package

## Recommended immediate submission set

- main paper PDF
- supplementary appendix PDF
- final mirrored key artifacts under `aaai2027/final/`
- if needed later:
  - one anonymized ZIP that packages code/data pointers and key reviewed artifacts

Current local convenience bundle:

- [aaai27_submission_bundle_current.zip](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/final/aaai27_submission_bundle_current.zip)
