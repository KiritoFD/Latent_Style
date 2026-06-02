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

## Current files

- `ai_research_prompt_usage_20260602.md`
  - records how external writing-support prompts were used during the current
    paper rewrite cycle

## Naming rule

Use date-stamped filenames so the literature audit remains append-only and easy
to diff:

- `related_work_audit_YYYYMMDD.md`
- `citation_gap_scan_YYYYMMDD.md`
- `paper_positioning_note_YYYYMMDD.md`
