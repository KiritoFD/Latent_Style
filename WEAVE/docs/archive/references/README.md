# Reference Audit Surface

This directory is the stable home for literature-facing support artifacts for
the AAAI 2027 paper push.

## What belongs here

- related-work audit memos
- citation coverage checks
- paper-positioning notes
- prompt or workflow notes used specifically for literature review and writing

## What does not belong here

- raw PDFs of papers
- experiment logs
- generated figures
- main manuscript text

Those belong in external paper managers, `docs/experiments/`,
`aaai_submission/figures/`, or `aaai_submission/`.

## Current high-value notes

- `literature_intel_memo_20260603.md`
  - broad current-paper literature audit and positioning snapshot
- `related_work_gap_candidates_20260603.md`
  - candidate gap list before later narrowing passes
- `related_work_and_intro_gap_recheck_20260603.md`
  - bounded reread of intro and related-work framing gaps
- `related_work_framing_patch_priorities_20260603.md`
  - current shortest list of literature/framing repairs still worth patching
- `tokenizer_representation_related_work_refresh_20260603.md`
  - tokenizer terminology and adjacent-method boundary note
- `evaluation_pathology_noop_memo_20260603.md`
  - metric-pathology framing support for the `idt` / no-op diagnosis
- `ai_research_prompt_usage_20260602.md`
  - records how external writing-support prompts were used during the current
    paper rewrite cycle

## Naming rule

Use date-stamped filenames so the literature audit remains append-only and easy
to diff:

- `related_work_audit_YYYYMMDD.md`
- `citation_gap_scan_YYYYMMDD.md`
- `paper_positioning_note_YYYYMMDD.md`
