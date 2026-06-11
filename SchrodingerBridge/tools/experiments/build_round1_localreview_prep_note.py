from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fmt_epochs(rows: list[dict[str, str]]) -> str:
    epochs: list[str] = []
    for row in rows:
        epoch = str(row.get("epoch", "")).strip()
        if epoch and epoch not in epochs:
            epochs.append(epoch)
    return ", ".join(epochs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a markdown prep note for round1 image-backed localreview from a fast bestfew handoff CSV.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--handoff-csv", type=Path, required=True)
    parser.add_argument("--output-note", type=Path, required=True)
    parser.add_argument("--review-local-root", type=Path, required=True)
    parser.add_argument("--review-eval-subdir", default="full_eval_fresh_localreview")
    parser.add_argument("--use-remote-rerun", action="store_true")
    args = parser.parse_args()

    handoff_csv = Path(args.handoff_csv).resolve()
    rows = _read_rows(handoff_csv)
    if not rows:
        raise RuntimeError(f"Empty handoff csv: {handoff_csv}")

    config_path = Path(args.config)
    review_local_root = Path(args.review_local_root).resolve()
    review_eval_subdir = str(args.review_eval_subdir).strip() or "full_eval_fresh_localreview"

    rerun_cmd = (
        f"python SchrodingerBridge\\tools\\experiments\\run_round1_family_bestfew_pipeline.py "
        f"--config {config_path} "
        f"--fast-local-root {handoff_csv.parent} "
        f"--fast-eval-subdir {handoff_csv.stem.replace('_bestfew_handoff', '')} "
        f"--review-local-root {review_local_root} "
        f"--review-eval-subdir {review_eval_subdir} "
        + ("--use-remote-rerun " if bool(args.use_remote_rerun) else "")
        + "--skip-introstyle --skip-dino"
    ).strip()

    review_cmd = (
        f"python SchrodingerBridge\\tools\\experiments\\run_local_round1_family_review.py "
        f"--config {config_path} "
        f"--eval-subdir {review_eval_subdir} "
        f"--local-root {review_local_root}"
    )

    lines = [
        "# Round1 Localreview Prep",
        "",
        "Purpose:",
        "",
        "- record the current image-backed deep-review prep state for this family",
        "- avoid re-deriving which checkpoints must be rerun with images before IntroStyle / DINO / VLM",
        "",
        "## Current Fast Bestfew",
        "",
        f"- Handoff CSV: `{handoff_csv}`",
        f"- Canonical epochs: `{_fmt_epochs(rows)}`",
        "- Reasons:",
    ]
    for row in rows:
        lines.append(f"  - `{str(row.get('reason', '')).strip()} -> {str(row.get('epoch', '')).strip()}`")
    lines.extend(
        [
            "",
            "## Next Commands",
            "",
            "- Build image-backed rerun packet:",
            f"  - `{rerun_cmd}`",
            "- Run local IntroStyle / DINO after image-backed rerun exists:",
            f"  - `{review_cmd}`",
            "",
            "## Notes",
            "",
            f"- Review output root: `{review_local_root}`",
            f"- Review eval subdir: `{review_eval_subdir}`",
            "- This note is intentionally command-oriented so the next deep-review handoff is fast and repeatable.",
        ]
    )

    output_note = Path(args.output_note).resolve()
    output_note.parent.mkdir(parents=True, exist_ok=True)
    output_note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
