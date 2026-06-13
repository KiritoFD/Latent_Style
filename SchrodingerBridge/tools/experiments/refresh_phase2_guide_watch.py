from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


KEYWORD_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("structure_breakthrough", ("topogate", "appalign", "lpips", "tradeoff")),
    ("style_lift_options", ("w_kinetic", "residual_gain", "style_spatial_pre_gain_16", "proximal_residual_energy_weight")),
    ("i2sb_path", ("i2sb", "sigma0.02", "sde", "brownian")),
    ("structure_backups", ("pnp", "self-inject", "pc solver", "solver_pc")),
    ("tokenizer_read", ("tokenizer", "query_dim", "query_num_blocks", "num_clusters", "attn_entropy", "attn_max", "revert")),
    ("cleanup", ("cleanup", "ckpt", "formal lane", "immortal")),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_grouped_hits(text: str) -> dict[str, list[dict[str, str]]]:
    lines = text.splitlines()
    groups: dict[str, list[dict[str, str]]] = {name: [] for name, _ in KEYWORD_GROUPS}
    heading = ""
    seen: set[tuple[str, str]] = set()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line
            continue
        lower = line.lower()
        for group_name, keywords in KEYWORD_GROUPS:
            if any(keyword in lower for keyword in keywords):
                key = (group_name, line)
                if key in seen:
                    continue
                seen.add(key)
                groups[group_name].append(
                    {
                        "heading": heading,
                        "line": line,
                    }
                )
                break
    return groups


def _render_status_md(
    *,
    guide_path: Path,
    guide_hash: str,
    last_hash: str,
    grouped_hits: dict[str, list[dict[str, str]]],
) -> str:
    changed = guide_hash != last_hash and bool(last_hash)
    lines: list[str] = [
        "# Phase2 Guide Watch Status",
        "",
        f"- Refreshed at: `{_utc_now_iso()}`",
        f"- Guide path: `{guide_path}`",
        f"- Guide sha256: `{guide_hash}`",
        f"- Guide changed since last run: `{changed}`",
        "",
        "## Current Read",
        "- This watcher is a low-noise local digest for `docs/612-phase2/guide_for_running_codex.md`.",
        "- It does not replace `docs/612-phase2/README.md`; it keeps the other model's actionable hints visible between Codex sessions.",
        "",
    ]
    for group_name, hits in grouped_hits.items():
        title = group_name.replace("_", " ").title()
        lines.append(f"## {title}")
        if not hits:
            lines.append("- n/a")
            lines.append("")
            continue
        for item in hits[:8]:
            heading = item.get("heading", "").strip()
            line = item.get("line", "").strip()
            if heading:
                lines.append(f"- {heading}: {line}")
            else:
                lines.append(f"- {line}")
        lines.append("")
    lines.extend(
        [
            "## Operating Rule",
            "- Prefer adopting guide suggestions that preserve the current in-band structure recovery.",
            "- Treat guide suggestions as advisory until they are reconciled with the live `phase2` curve and queue rules.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh a low-noise digest for docs/612-phase2/guide_for_running_codex.md.")
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--status-md", type=Path, required=True)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--history-jsonl", type=Path, required=True)
    args = parser.parse_args()

    guide_path = Path(args.guide).expanduser().resolve()
    status_md = Path(args.status_md).expanduser().resolve()
    state_json = Path(args.state_json).expanduser().resolve()
    history_jsonl = Path(args.history_jsonl).expanduser().resolve()

    text = guide_path.read_text(encoding="utf-8", errors="replace")
    guide_hash = _sha256_text(text)
    last_hash = ""
    if state_json.is_file():
        try:
            prev = json.loads(state_json.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                last_hash = str(prev.get("guide_hash", "")).strip()
        except Exception:
            last_hash = ""

    grouped_hits = _extract_grouped_hits(text)
    status_md.parent.mkdir(parents=True, exist_ok=True)
    state_json.parent.mkdir(parents=True, exist_ok=True)
    history_jsonl.parent.mkdir(parents=True, exist_ok=True)
    status_md.write_text(
        _render_status_md(
            guide_path=guide_path,
            guide_hash=guide_hash,
            last_hash=last_hash,
            grouped_hits=grouped_hits,
        ),
        encoding="utf-8",
    )

    state_payload = {
        "refreshed_at": _utc_now_iso(),
        "guide_path": str(guide_path),
        "guide_hash": guide_hash,
        "guide_changed": bool(last_hash and last_hash != guide_hash),
        "group_counts": {name: len(items) for name, items in grouped_hits.items()},
        "status_md": str(status_md),
    }
    state_json.write_text(json.dumps(state_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if guide_hash != last_hash:
        event = dict(state_payload)
        with history_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(status_md)
    print(state_json)
    print(history_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
