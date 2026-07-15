from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


AAAI_ROOT = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027")
DEST_ROOT_NAME = "results_snapshot_20260611"

SKIP_NAMES = {
    ".gitignore",
    "ACTIVE_DRAFT.md",
    "FORMAT_NOTES.md",
    "aaai2026.bst",
    "aaai2026.sty",
    "aaai2027.bib",
    "aaai2027.bst",
    "aaai2027.sty",
    "build_paper.bat",
    "build_paper_improved.bat",
    "build_paper_with_svg.bat",
    "build_supplement.bat",
    "figures_config.py",
    "local_analysis_environment.yml",
    "local_analysis_requirements.txt",
    "official_requirement_check.md",
    "round1_newdata_variant_board.csv",
}

SKIP_SUFFIXES = (".py", ".bat", ".sty", ".bst", ".bib", ".tex", ".pdf", ".png", ".svg", ".tar", ".pt", ".pth", ".b64")

ACTIVE_LOG_PREFIXES = (
    "samst_wikiarts5_",
)


def _category_for(path: Path) -> str | None:
    name = path.name
    if name in SKIP_NAMES:
        return None
    if name.startswith(ACTIVE_LOG_PREFIXES):
        return None
    group = "misc"
    lower = name.lower()
    if lower.startswith("vlm_"):
        group = "vlm"
    elif lower.startswith("round1_") or lower.startswith("_round1_") or lower.startswith("_packet_smoke_") or lower.startswith("_stageclose_smoke_"):
        group = "round1"
    elif any(token in lower for token in ("introstyle", "dino", "bestfew", "local_finalists", "dualpath", "edgegated", "hold4twostage", "knee_")):
        group = "introstyle_dino"
    elif any(token in lower for token in ("samst", "samam", "wikiarts5", "distinct5", "operating_point")):
        group = "baselines"
    elif lower.startswith("tmp_") or lower.startswith("paper_") or lower.startswith("bibtex") or "claim_" in lower:
        group = "paper_audit"
    if path.suffix.lower() in {".csv"}:
        return f"{group}/csv"
    if path.suffix.lower() in {".jsonl"}:
        return f"{group}/jsonl"
    if path.suffix.lower() in {".json"}:
        return f"{group}/json"
    if path.suffix.lower() in {".txt"}:
        return f"{group}/text"
    if name.endswith((".stdout.log", ".stderr.log", ".err.log", ".log")):
        return f"{group}/logs"
    if name.endswith(SKIP_SUFFIXES):
        return None
    return None


def _iter_loose_files(root: Path) -> list[Path]:
    return [path for path in sorted(root.iterdir()) if path.is_file()]


def _write_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "original_name",
                "original_path",
                "moved_path",
                "size_bytes",
                "modified_utc",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path, *, moved_rows: list[dict[str, str]]) -> None:
    counts = Counter(row["category"] for row in moved_rows)
    lines = [
        "# Results Snapshot 2026-06-11",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Source root: `{AAAI_ROOT}`",
        f"- Total moved files: `{len(moved_rows)}`",
        "",
        "## Counts",
        "",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- `{category}`: `{count}`")
    lines += [
        "",
        "## Contract",
        "",
        "- Root-level loose result files are moved under this snapshot by topic and file type.",
        "- `index.csv` is the machine-readable manifest.",
        "- Active live `SaMST` watcher logs are intentionally left outside this snapshot while training is running.",
        "",
        "## Current Summary",
        "",
        "- `SaMST` is no longer stuck at the first public eval point; the segmented auto-resume controller is active and is currently advancing the common frontier from `epoch_0005` to `epoch_0010`.",
        "- The wikiarts5 new-data variant board has been stabilized onto a fixed point CSV, so plotting and annotation edits now come from one source of truth.",
        "- The main cleanup target in this pass is aaai2027 root clutter: small CSV / log / JSON / JSONL / TXT files are being consolidated into this timestamped snapshot instead of remaining loose at root.",
        "",
        "## Current Conclusion",
        "",
        "- Keep experiment evidence, but stop leaving it flat at aaai2027 root.",
        "- Preserve live runs and formal result packets; archive or delete only temporary scripts and disposable scratch artifacts after they are indexed.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Move loose aaai2027 generated files into a typed artifact directory with an index.")
    parser.add_argument("--root", type=Path, default=AAAI_ROOT)
    parser.add_argument("--dest-name", default=DEST_ROOT_NAME)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    dest_root = root / str(args.dest_name)
    moved_rows: list[dict[str, str]] = []
    for path in _iter_loose_files(root):
        category = _category_for(path)
        if category is None:
            continue
        dest_dir = dest_root / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / path.name
        if dest_path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_path = dest_dir / f"{path.stem}_{stamp}{path.suffix}"
        stat = path.stat()
        try:
            shutil.move(str(path), str(dest_path))
            status = "moved"
            moved_path = str(dest_path)
        except PermissionError:
            status = "locked_skip"
            moved_path = ""
        moved_rows.append(
            {
                "category": category,
                "original_name": path.name,
                "original_path": str(path),
                "moved_path": moved_path,
                "size_bytes": str(stat.st_size),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "status": status,
            }
        )

    _write_index(dest_root / "index.csv", moved_rows)
    _write_readme(dest_root / "README.md", moved_rows=moved_rows)
    print(dest_root)
    print(f"moved={len(moved_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
