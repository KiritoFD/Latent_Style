from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


KEYWORD_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("structure_breakthrough", ("topogate", "appalign", "lpips", "tradeoff")),
    ("style_lift_options", ("w_kinetic", "residual_gain", "style_spatial_pre_gain_16", "proximal_residual_energy_weight", "semantic_self_topology_blend", "topology_blend")),
    ("i2sb_path", ("i2sb", "sigma0.02", "sde", "sde-em", "stochastic flow matching", "brownian", "endpoint", "schrödinger bridge")),
    ("structure_backups", ("pnp", "self-inject", "pc solver", "solver_pc", "predictor-corrector", "langevin", "euler-maruyama", "solver_unsb_cycle")),
    ("tokenizer_read", ("tokenizer", "query_dim", "query_num_blocks", "num_clusters", "spatial_dim", "attn_entropy", "attn_max", "positional encoding", "sinusoidal", "revert")),
    ("cleanup", ("cleanup", "ckpt", "formal lane", "immortal")),
]

GUIDE_TEXT_ENCODINGS: tuple[str, ...] = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_guide_text(path: Path) -> tuple[str, str, bytes]:
    raw = path.read_bytes()
    for encoding in GUIDE_TEXT_ENCODINGS:
        try:
            return raw.decode(encoding), encoding, raw
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace", raw


def _load_json_dict(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{str(k): str(v) for k, v in row.items()} for row in reader]


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


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


def _derive_live_overlay(snapshot: dict) -> dict[str, object]:
    resolved = snapshot.get("resolved_packets") if isinstance(snapshot.get("resolved_packets"), dict) else {}
    structure = resolved.get("structure_reentry") if isinstance(resolved, dict) and isinstance(resolved.get("structure_reentry"), dict) else {}
    i2sb = resolved.get("i2sb_diagnostic_only") if isinstance(resolved, dict) and isinstance(resolved.get("i2sb_diagnostic_only"), dict) else {}
    remote_structure = snapshot.get("remote_structure_status") if isinstance(snapshot.get("remote_structure_status"), dict) else {}
    remote_i2sb = snapshot.get("remote_i2sb_status") if isinstance(snapshot.get("remote_i2sb_status"), dict) else {}
    curve = remote_structure.get("curve_summary") if isinstance(remote_structure.get("curve_summary"), dict) else {}
    latest = curve.get("latest") if isinstance(curve.get("latest"), dict) else {}
    convergence = remote_structure.get("convergence") if isinstance(remote_structure.get("convergence"), dict) else {}
    pending_epochs = remote_structure.get("pending_checkpoint_epochs") if isinstance(remote_structure.get("pending_checkpoint_epochs"), list) else []

    transfer_style = _float_or_none(latest.get("transfer_clip_style"))
    transfer_lpips = _float_or_none(latest.get("transfer_content_lpips"))
    allpairs_style = _float_or_none(latest.get("all_pairs_clip_style"))
    allpairs_lpips = _float_or_none(latest.get("all_pairs_content_lpips"))
    latest_epoch_int = int(latest.get("epoch_int", -1)) if str(latest.get("epoch_int", "")).strip() else -1

    style_limited = bool(
        transfer_style is not None
        and transfer_style <= 0.68
        and allpairs_lpips is not None
        and allpairs_lpips <= 0.32
    )
    recovered_structure = bool(
        allpairs_style is not None
        and allpairs_style >= 0.701666
        and allpairs_lpips is not None
        and allpairs_lpips <= 0.381724
    )
    min_settled_epoch = int(str(structure.get("watch_min_settled_epoch", "")).strip() or 0)
    min_epoch_met = latest_epoch_int >= min_settled_epoch if min_settled_epoch > 0 else True
    best_in_newest_2 = bool(convergence.get("best_in_newest_2")) if isinstance(convergence, dict) else False
    tail_flat = bool(convergence.get("tail_flat")) if isinstance(convergence, dict) else False
    close_gate_ready = bool(
        str(structure.get("packet_id", "")).strip() == "vel_tok32_safe_semantic_topogate_k085_appalign"
        and recovered_structure
        and style_limited
        and min_epoch_met
        and (not best_in_newest_2)
        and tail_flat
        and not pending_epochs
    )
    close_gate_blockers: list[str] = []
    if not min_epoch_met:
        close_gate_blockers.append(f"latest_settled_epoch<{min_settled_epoch}")
    if best_in_newest_2:
        close_gate_blockers.append("best_in_newest_2=true")
    if not tail_flat:
        close_gate_blockers.append("tail_flat=false")
    if pending_epochs:
        close_gate_blockers.append("pending_checkpoint_epochs")

    i2sb_status = str(i2sb.get("status", "")).strip().lower()
    i2sb_live_state = str(remote_i2sb.get("live_state", "")).strip().lower()
    i2sb_is_active = i2sb_status in {"running", "launch_requested"} or i2sb_live_state in {
        "training_before_first_settled_eval",
        "training_after_settled_eval",
        "eval_in_progress_or_pending",
    }

    if i2sb_is_active:
        recommendation = "monitor_running_i2sb_sigma0p02_tfloor005"
        secondary = "after_first_i2sb_settled_read_reassess_pc_solver_need"
    elif close_gate_ready:
        recommendation = "launch_i2sb_sigma0p02_tfloor005_now"
        secondary = "after_i2sb_read_if_still_needed_run_eval_only_pc_solver"
    elif recovered_structure and style_limited:
        recommendation = "continue_appalign_until_close_gate"
        secondary = "keep_i2sb_sigma0p02_tfloor005_as_next_diagnostic"
    else:
        recommendation = "continue_current_structure_lane"
        secondary = "do_not_queue_style_lift_branch_yet"

    return {
        "structure_packet": str(structure.get("packet_id", "")).strip(),
        "structure_watch_handoff_mode": str(structure.get("watch_handoff_mode", "")).strip(),
        "latest_settled_epoch": str(latest.get("epoch", "")).strip(),
        "latest_transfer_clip_style": transfer_style,
        "latest_transfer_content_lpips": transfer_lpips,
        "latest_all_pairs_clip_style": allpairs_style,
        "latest_all_pairs_content_lpips": allpairs_lpips,
        "style_limited_under_recovered_structure": style_limited,
        "recovered_structure_band": recovered_structure,
        "close_gate_ready": close_gate_ready,
        "close_gate_blockers": close_gate_blockers,
        "recommended_next_action": recommendation,
        "recommended_followup_after_next_action": secondary,
        "preferred_i2sb_packet": str(i2sb.get("packet_id", "")).strip(),
        "i2sb_live_state": i2sb_live_state,
    }


def _derive_reconciliation(snapshot: dict) -> list[str]:
    lines: list[str] = []
    resolved = snapshot.get("resolved_packets") if isinstance(snapshot.get("resolved_packets"), dict) else {}
    structure = resolved.get("structure_reentry") if isinstance(resolved.get("structure_reentry"), dict) else {}
    i2sb = resolved.get("i2sb_diagnostic_only") if isinstance(resolved.get("i2sb_diagnostic_only"), dict) else {}
    structure_packet = str(structure.get("packet_id", "")).strip()
    i2sb_status = str(i2sb.get("status", "")).strip().lower()

    if structure_packet == "vel_tok32_safe_semantic_topogate_k070":
        lines.append("Guide proposal `semantic_self_topology_blend=0.7` has already been adopted as the current live structure lane (`k070`).")
    if i2sb_status.startswith("closed"):
        lines.append("Guide proposal `I2SB σ=0.02` has already been executed and is currently classified as archival-only negative evidence.")

    pc_summary = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\phase2_pc_eval_appalign_e3\summary.json")
    if pc_summary.is_file():
        lines.append("Guide proposal `solver_pc + latent_lowpass` has already been run as a side probe and is currently treated as negative evidence.")

    manifest_rows = _load_manifest_rows(Path(str(snapshot.get("manifest_csv", "")).strip()))
    for row in manifest_rows:
        if str(row.get("packet_id", "")).strip() == "vel_tok32_safe_semantic_topogate_k070_sp256":
            lines.append("The remaining low-risk tokenizer suggestion has been absorbed as queued follow-on `k070_sp256` via `tokenizer_spatial_dim=256`.")
            break
    return lines


def _render_status_md(
    *,
    guide_path: Path,
    guide_hash: str,
    guide_encoding: str,
    last_hash: str,
    grouped_hits: dict[str, list[dict[str, str]]],
    live_overlay: dict[str, object] | None,
    reconciliation: list[str] | None,
) -> str:
    changed = guide_hash != last_hash and bool(last_hash)
    lines: list[str] = [
        "# Phase2 Guide Watch Status",
        "",
        f"- Refreshed at: `{_utc_now_iso()}`",
        f"- Guide path: `{guide_path}`",
        f"- Guide sha256: `{guide_hash}`",
        f"- Guide decoding: `{guide_encoding}`",
        f"- Guide changed since last run: `{changed}`",
        "",
        "## Current Read",
        "- This watcher is a low-noise local digest for `docs/612-phase2/guide_for_running_codex.md`.",
        "- It does not replace `docs/612-phase2/README.md`; it keeps the other model's actionable hints visible between Codex sessions.",
        "",
    ]
    if isinstance(live_overlay, dict) and live_overlay:
        lines.extend(
            [
                "## Live Decision Overlay",
                f"- Structure packet: `{str(live_overlay.get('structure_packet', 'n/a'))}`",
                f"- Latest settled epoch: `{str(live_overlay.get('latest_settled_epoch', 'n/a'))}`",
                f"- Transfer `CLIP-S / LPIPS`: `{_float_or_none(live_overlay.get('latest_transfer_clip_style')) or 0.0:.6f} / {(_float_or_none(live_overlay.get('latest_transfer_content_lpips')) or 0.0):.6f}`" if _float_or_none(live_overlay.get("latest_transfer_clip_style")) is not None and _float_or_none(live_overlay.get("latest_transfer_content_lpips")) is not None else "- Transfer `CLIP-S / LPIPS`: n/a",
                f"- All-pairs `CLIP-S / LPIPS`: `{_float_or_none(live_overlay.get('latest_all_pairs_clip_style')) or 0.0:.6f} / {(_float_or_none(live_overlay.get('latest_all_pairs_content_lpips')) or 0.0):.6f}`" if _float_or_none(live_overlay.get("latest_all_pairs_clip_style")) is not None and _float_or_none(live_overlay.get("latest_all_pairs_content_lpips")) is not None else "- All-pairs `CLIP-S / LPIPS`: n/a",
                f"- Style-limited under recovered structure: `{bool(live_overlay.get('style_limited_under_recovered_structure', False))}`",
                f"- Close gate ready: `{bool(live_overlay.get('close_gate_ready', False))}`",
                (
                    f"- Close gate blockers: `{', '.join(str(x) for x in (live_overlay.get('close_gate_blockers') or []))}`"
                    if live_overlay.get("close_gate_blockers")
                    else "- Close gate blockers: `none`"
                ),
                f"- Recommended next action: `{str(live_overlay.get('recommended_next_action', 'n/a'))}`",
                f"- Recommended follow-up: `{str(live_overlay.get('recommended_followup_after_next_action', 'n/a'))}`",
                f"- I2SB live state: `{str(live_overlay.get('i2sb_live_state', 'n/a'))}`",
                "",
            ]
        )
    if reconciliation:
        lines.extend(
            [
                "## Evidence Reconciliation",
                *[f"- {line}" for line in reconciliation],
                "",
            ]
        )
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
    parser.add_argument("--phase2-snapshot", type=Path, default=None)
    args = parser.parse_args()

    guide_path = Path(args.guide).expanduser().resolve()
    status_md = Path(args.status_md).expanduser().resolve()
    state_json = Path(args.state_json).expanduser().resolve()
    history_jsonl = Path(args.history_jsonl).expanduser().resolve()
    phase2_snapshot = Path(args.phase2_snapshot).expanduser().resolve() if args.phase2_snapshot else None

    text, guide_encoding, guide_bytes = _load_guide_text(guide_path)
    guide_hash = _sha256_bytes(guide_bytes)
    last_hash = ""
    if state_json.is_file():
        try:
            prev = json.loads(state_json.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                last_hash = str(prev.get("guide_hash", "")).strip()
        except Exception:
            last_hash = ""

    grouped_hits = _extract_grouped_hits(text)
    live_overlay = {}
    reconciliation: list[str] = []
    if phase2_snapshot is not None and phase2_snapshot.is_file():
        try:
            snapshot = _load_json_dict(phase2_snapshot)
            live_overlay = _derive_live_overlay(snapshot)
            reconciliation = _derive_reconciliation(snapshot)
        except Exception:
            live_overlay = {}
            reconciliation = []
    status_md.parent.mkdir(parents=True, exist_ok=True)
    state_json.parent.mkdir(parents=True, exist_ok=True)
    history_jsonl.parent.mkdir(parents=True, exist_ok=True)
    status_md.write_text(
        _render_status_md(
            guide_path=guide_path,
            guide_hash=guide_hash,
            guide_encoding=guide_encoding,
            last_hash=last_hash,
            grouped_hits=grouped_hits,
            live_overlay=live_overlay,
            reconciliation=reconciliation,
        ),
        encoding="utf-8-sig",
    )

    state_payload = {
        "refreshed_at": _utc_now_iso(),
        "guide_path": str(guide_path),
        "guide_hash": guide_hash,
        "guide_encoding": guide_encoding,
        "guide_changed": bool(last_hash and last_hash != guide_hash),
        "group_counts": {name: len(items) for name, items in grouped_hits.items()},
        "status_md": str(status_md),
        "phase2_snapshot": str(phase2_snapshot) if phase2_snapshot is not None else "",
        "live_overlay": live_overlay,
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
