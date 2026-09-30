from __future__ import annotations

import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SNAPSHOT = SB_ROOT / "docs" / "experiments" / "phase2_queue_state_snapshot.json"
DEFAULT_OUTPUT = SB_ROOT / "docs" / "experiments" / "phase2_current_status.md"
DEFAULT_DISTINCT5_IDT_SUMMARY = (
    SB_ROOT / "docs" / "experiments" / "idt_eval_20260602" / "distinct5_512" / "idt_5x5" / "summary.json"
)


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _path_link(path_like: str, label: str | None = None) -> str:
    raw = str(path_like).strip()
    if not raw:
        return ""
    display = label or Path(raw).name or raw
    normalized = raw.replace("\\", "/")
    if normalized[1:3] == ":/":
        normalized = "/" + normalized
    return f"[{display}]({normalized})"


def _fmt_float(value: object, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _fmt_sec(value: object) -> str:
    try:
        return f"{float(value):.2f}s"
    except Exception:
        return "n/a"


def _fmt_mib(used: object, total: object) -> str:
    try:
        return f"{int(float(used))} / {int(float(total))} MiB"
    except Exception:
        return "n/a"


def _epoch_int(row: dict[str, object]) -> int:
    try:
        return int(row.get("epoch_int", 0))
    except Exception:
        return 0


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _recovery_read(
    *,
    row: dict[str, object],
    style_key: str,
    lpips_key: str,
    style_target: str,
    lpips_target: str,
    min_epoch: str,
) -> str:
    style = _float_or_none(row.get(style_key))
    lpips = _float_or_none(row.get(lpips_key))
    target_style = _float_or_none(style_target)
    target_lpips = _float_or_none(lpips_target)
    try:
        min_epoch_int = int(min_epoch)
    except Exception:
        min_epoch_int = 0
    if style is None or lpips is None or target_style is None or target_lpips is None:
        return "n/a"
    if _epoch_int(row) < min_epoch_int:
        return f"not eligible before settled epoch {min_epoch_int}"
    style_gap = style - target_style
    lpips_margin = target_lpips - lpips
    if style_gap >= 0.0 and lpips_margin >= 0.0:
        return (
            "recovered: "
            f"style +{style_gap:.6f}, LPIPS margin +{lpips_margin:.6f}"
        )
    parts: list[str] = []
    if style_gap < 0.0:
        parts.append(f"style short by {abs(style_gap):.6f}")
    else:
        parts.append(f"style +{style_gap:.6f}")
    if lpips_margin < 0.0:
        parts.append(f"LPIPS over by {abs(lpips_margin):.6f}")
    else:
        parts.append(f"LPIPS margin +{lpips_margin:.6f}")
    return ", ".join(parts)


def _load_distinct5_idt_style_refs() -> dict[str, float]:
    path = DEFAULT_DISTINCT5_IDT_SUMMARY
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    analysis = payload.get("analysis") if isinstance(payload, dict) else None
    if not isinstance(analysis, dict):
        return {}
    refs: dict[str, float] = {}
    try:
        refs["all_pairs_clip_style"] = float(analysis["all_pairs_overview"]["clip_style"])
        refs["transfer_clip_style"] = float(analysis["style_transfer_ability"]["clip_style"])
    except Exception:
        return {}
    return refs


def _style_delta_vs_idt(value: object, ref: float | None) -> str:
    metric = _float_or_none(value)
    if metric is None or ref is None:
        return "n/a"
    return f"{metric - ref:+.6f}"


def _runtime_observability_brief(summary_like: object) -> str:
    if not isinstance(summary_like, dict):
        return "n/a"
    runtime = summary_like.get("runtime_observability")
    if not isinstance(runtime, dict) or not bool(runtime.get("available")):
        return "n/a"
    preferred = runtime.get("style_transfer_ability")
    if not isinstance(preferred, dict) or not preferred:
        preferred = runtime.get("all_pairs_overview")
    if not isinstance(preferred, dict) or not preferred:
        return "n/a"
    parts: list[str] = []
    display_keys = [
        ("structured_style_tokenizer_attn_effective_count", "tok_eff", 1),
        ("structured_style_tokenizer_gate_mean", "gate", 3),
        ("structured_style_tokenizer_mask_mean", "mask", 3),
        ("semantic_topology_attn_entropy", "topo_ent", 3),
        ("output_appearance_active", "app_on", 1),
        ("output_appearance_scale_mean", "app_s", 3),
        ("output_appearance_shift_abs", "app_d", 3),
        ("i2sb_time_floor", "t_floor", 3),
        ("i2sb_time_floor_active", "t_floor_on", 1),
        ("i2sb_predict_t", "pred_t", 3),
    ]
    for key, label, digits in display_keys:
        value = _float_or_none(preferred.get(key))
        if value is None:
            continue
        parts.append(f"{label}={value:.{digits}f}")
    return ", ".join(parts) if parts else "n/a"


def _train_metrics_observability_brief(metrics_like: object) -> str:
    if not isinstance(metrics_like, dict):
        return "n/a"
    parts: list[str] = []
    display_keys = [
        ("epoch", "train_ep", 0),
        ("structured_style_tokenizer_attn_effective_count", "tok_eff", 1),
        ("structured_style_tokenizer_gate_mean", "gate", 3),
        ("structured_style_tokenizer_mask_mean", "mask", 3),
        ("semantic_topology_attn_entropy", "topo_ent", 3),
        ("output_appearance_active", "app_on", 1),
        ("output_appearance_scale_mean", "app_s", 3),
        ("output_appearance_shift_abs", "app_d", 3),
        ("bridge_noise_schedule_exact", "bex", 0),
        ("bridge_sigma", "sigma", 3),
    ]
    for key, label, digits in display_keys:
        value = _float_or_none(metrics_like.get(key))
        if value is None:
            continue
        parts.append(f"{label}={value:.{digits}f}")
    return ", ".join(parts) if parts else "n/a"


def _best_observability_brief(summary_like: object, train_metrics_like: object) -> str:
    summary_brief = _runtime_observability_brief(summary_like)
    if summary_brief != "n/a":
        return summary_brief
    return _train_metrics_observability_brief(train_metrics_like)


def build_note(snapshot: dict, *, report_date: str) -> str:
    resolved = snapshot.get("resolved_packets", {}) if isinstance(snapshot.get("resolved_packets"), dict) else {}
    formal = resolved.get("formal_lane", {}) if isinstance(resolved.get("formal_lane"), dict) else {}
    structure = resolved.get("structure_reentry", {}) if isinstance(resolved.get("structure_reentry"), dict) else {}
    i2sb = resolved.get("i2sb_diagnostic_only", {}) if isinstance(resolved.get("i2sb_diagnostic_only"), dict) else {}
    remote = snapshot.get("remote_formal_status", {}) if isinstance(snapshot.get("remote_formal_status"), dict) else {}
    remote_structure = snapshot.get("remote_structure_status", {}) if isinstance(snapshot.get("remote_structure_status"), dict) else {}
    remote_i2sb = snapshot.get("remote_i2sb_status", {}) if isinstance(snapshot.get("remote_i2sb_status"), dict) else {}
    health = snapshot.get("remote_health", {}) if isinstance(snapshot.get("remote_health"), dict) else {}
    local_velocity_watchers = snapshot.get("local_velocity_handoff_watchers", []) if isinstance(snapshot.get("local_velocity_handoff_watchers"), list) else []
    local_structure_logs = snapshot.get("local_structure_watcher_logs", {}) if isinstance(snapshot.get("local_structure_watcher_logs"), dict) else {}
    idt_refs = _load_distinct5_idt_style_refs()
    curve = remote.get("curve_summary", {}) if isinstance(remote.get("curve_summary"), dict) else {}
    latest = curve.get("latest", {}) if isinstance(curve.get("latest"), dict) else {}
    best_transfer = curve.get("best_transfer", {}) if isinstance(curve.get("best_transfer"), dict) else {}
    best_all_pairs = curve.get("best_all_pairs", {}) if isinstance(curve.get("best_all_pairs"), dict) else {}
    structure_curve = remote_structure.get("curve_summary", {}) if isinstance(remote_structure.get("curve_summary"), dict) else {}
    structure_latest = structure_curve.get("latest", {}) if isinstance(structure_curve.get("latest"), dict) else {}
    gpus = remote.get("remote_gpu", []) if isinstance(remote.get("remote_gpu"), list) else []
    gpu0 = gpus[0] if gpus and isinstance(gpus[0], dict) else {}
    formal_status_text = str(formal.get("status", "n/a"))
    formal_live_state = str(remote.get("live_state", "n/a"))
    formal_processes = remote.get("processes", []) if isinstance(remote.get("processes"), list) else []
    if formal_status_text.startswith("closed"):
        formal_live_state = "settled_no_live_process"
        gpu0 = {}
    structure_status_text = str(structure.get("status", "n/a"))
    structure_live_state = str(remote_structure.get("live_state", "n/a"))
    if structure_status_text not in {"running", "launch_requested"}:
        structure_live_state = "n/a"
    i2sb_status_text = str(i2sb.get("status", "n/a"))
    i2sb_live_state = str(remote_i2sb.get("live_state", "n/a"))
    if i2sb_status_text not in {"running", "launch_requested"}:
        i2sb_live_state = "n/a"

    snapshot_link = _path_link(str(snapshot.get("manifest_csv", "")), "phase2_queue_manifest.csv")
    validation_link = _path_link(str(snapshot.get("validation_json", "")), "phase2_queue_manifest_validation.json")
    state_link = _path_link(str(DEFAULT_SNAPSHOT.resolve()), "phase2_queue_state_snapshot.json")
    formal_cfg = _path_link(str(formal.get("config_path", "")), "formal config")
    formal_note = _path_link(str(formal.get("note_path", "")), "formal note")
    structure_cfg = _path_link(str(structure.get("config_path", "")), "structure config")
    structure_note = _path_link(str(structure.get("note_path", "")), "structure note")
    i2sb_cfg = _path_link(str(i2sb.get("config_path", "")), "I2SB config")
    i2sb_note = _path_link(str(i2sb.get("note_path", "")), "I2SB note")

    latest_transfer = (
        f"{_fmt_float(latest.get('transfer_clip_style'))} / "
        f"{_fmt_float(latest.get('transfer_content_lpips'))}"
    )
    latest_all_pairs = (
        f"{_fmt_float(latest.get('all_pairs_clip_style'))} / "
        f"{_fmt_float(latest.get('all_pairs_content_lpips'))}"
    )
    best_transfer_pair = (
        f"{_fmt_float(best_transfer.get('transfer_clip_style'))} / "
        f"{_fmt_float(best_transfer.get('transfer_content_lpips'))}"
    )
    best_all_pairs_pair = (
        f"{_fmt_float(best_all_pairs.get('all_pairs_clip_style'))} / "
        f"{_fmt_float(best_all_pairs.get('all_pairs_content_lpips'))}"
    )

    latest_allpairs_recovery = _recovery_read(
        row=latest,
        style_key="all_pairs_clip_style",
        lpips_key="all_pairs_content_lpips",
        style_target=str(formal.get("watch_min_allpairs_style_recovery", "")),
        lpips_target=str(formal.get("watch_max_allpairs_lpips_for_recovery", "")),
        min_epoch=str(formal.get("watch_min_settled_epoch", "")),
    )
    latest_transfer_recovery = _recovery_read(
        row=latest,
        style_key="transfer_clip_style",
        lpips_key="transfer_content_lpips",
        style_target=str(formal.get("watch_min_transfer_style_recovery", "")),
        lpips_target=str(formal.get("watch_max_transfer_lpips_for_recovery", "")),
        min_epoch=str(formal.get("watch_min_settled_epoch", "")),
    )
    best_allpairs_recovery = _recovery_read(
        row=best_all_pairs,
        style_key="all_pairs_clip_style",
        lpips_key="all_pairs_content_lpips",
        style_target=str(formal.get("watch_min_allpairs_style_recovery", "")),
        lpips_target=str(formal.get("watch_max_allpairs_lpips_for_recovery", "")),
        min_epoch=str(formal.get("watch_min_settled_epoch", "")),
    )
    best_transfer_recovery = _recovery_read(
        row=best_transfer,
        style_key="transfer_clip_style",
        lpips_key="transfer_content_lpips",
        style_target=str(formal.get("watch_min_transfer_style_recovery", "")),
        lpips_target=str(formal.get("watch_max_transfer_lpips_for_recovery", "")),
        min_epoch=str(formal.get("watch_min_settled_epoch", "")),
    )

    lines = [
        "# Phase 2 Current Status",
        "",
        f"Date: {report_date}",
        "",
        "## Sources",
        f"- Queue manifest: {snapshot_link}" if snapshot_link else "- Queue manifest: n/a",
        f"- Validation snapshot: {validation_link}" if validation_link else "- Validation snapshot: n/a",
        f"- State snapshot: {state_link}" if state_link else "- State snapshot: n/a",
        "",
        "## Formal Lane",
        f"- Preferred packet: `{formal.get('packet_id', 'n/a')}`",
        f"- Status: `{formal_status_text}`",
        f"- Run: `{formal.get('run_name', 'n/a')}`",
        f"- Config: {formal_cfg}" if formal_cfg else "- Config: n/a",
        f"- Note: {formal_note}" if formal_note else "- Note: n/a",
        f"- Live state: `{formal_live_state}`",
        f"- Remote GPU: {_fmt_mib(gpu0.get('memory_used_mib'), gpu0.get('memory_total_mib')) if gpu0 else 'n/a'}",
        f"- Current read: {formal.get('current_read', 'n/a')}",
        "",
        "### Latest Settled Point",
        f"- Epoch: `{latest.get('epoch', 'n/a')}`",
        f"- Transfer `CLIP-S / LPIPS`: `{latest_transfer}`",
        f"- Transfer `style - IDT`: `{_style_delta_vs_idt(latest.get('transfer_clip_style'), idt_refs.get('transfer_clip_style'))}`",
        f"- All-pairs `CLIP-S / LPIPS`: `{latest_all_pairs}`",
        f"- All-pairs `style - IDT`: `{_style_delta_vs_idt(latest.get('all_pairs_clip_style'), idt_refs.get('all_pairs_clip_style'))}`",
        (
            f"- Identity `CLIP-S / LPIPS`: "
            f"`{_fmt_float(latest.get('identity_clip_style'))} / {_fmt_float(latest.get('identity_content_lpips'))}`"
        ),
        (
            f"- Eval timing: wall `{_fmt_sec(latest.get('eval_wall_total_sec'))}`, "
            f"eval `{_fmt_sec(latest.get('eval_total_sec'))}`, "
            f"generation `{_fmt_sec(latest.get('generation_sec'))}`, "
            f"decode `{_fmt_sec(latest.get('vae_decode_sec'))}`"
        ),
        "",
        "### Recovery Gate",
        f"- Min settled epoch: `{formal.get('watch_min_settled_epoch', 'n/a')}`",
        (
            f"- All-pairs target: style `>= {formal.get('watch_min_allpairs_style_recovery', 'n/a')}`, "
            f"LPIPS `<= {formal.get('watch_max_allpairs_lpips_for_recovery', 'n/a')}`"
        ),
        (
            f"- Transfer target: style `>= {formal.get('watch_min_transfer_style_recovery', 'n/a')}`, "
            f"LPIPS `<= {formal.get('watch_max_transfer_lpips_for_recovery', 'n/a')}`"
        ),
        f"- Latest all-pairs read: {latest_allpairs_recovery}",
        f"- Latest transfer read: {latest_transfer_recovery}",
        "",
        "### Best Settled Points In This Run",
        f"- Best transfer epoch: `{best_transfer.get('epoch', 'n/a')}` with `{best_transfer_pair}`",
        f"- Best transfer gate read: {best_transfer_recovery}",
        f"- Best all-pairs epoch: `{best_all_pairs.get('epoch', 'n/a')}` with `{best_all_pairs_pair}`",
        f"- Best all-pairs gate read: {best_allpairs_recovery}",
        "",
        "## Next Packets",
        f"- Structure-side preferred packet: `{structure.get('packet_id', 'n/a')}`",
        f"- Structure config: {structure_cfg}" if structure_cfg else "- Structure config: n/a",
        f"- Structure note: {structure_note}" if structure_note else "- Structure note: n/a",
        f"- Structure read: {structure.get('current_read', 'n/a')}",
        f"- Structure live state: `{structure_live_state}`",
        (
            f"- Structure GPU: {_fmt_mib((remote_structure.get('remote_gpu') or [{}])[0].get('memory_used_mib'), (remote_structure.get('remote_gpu') or [{}])[0].get('memory_total_mib'))}"
            if structure_status_text in {"running", "launch_requested"}
            and isinstance(remote_structure.get("remote_gpu"), list)
            and remote_structure.get("remote_gpu")
            else "- Structure GPU: n/a"
        ),
        f"- Structure latest settled epoch: `{remote_structure.get('latest_settled_epoch', '') or 'n/a'}`",
        (
            f"- Structure latest settled `CLIP-S / LPIPS`: "
            f"`{_fmt_float(structure_latest.get('transfer_clip_style'))} / {_fmt_float(structure_latest.get('transfer_content_lpips'))}`, "
            f"`{_fmt_float(structure_latest.get('all_pairs_clip_style'))} / {_fmt_float(structure_latest.get('all_pairs_content_lpips'))}`"
            if structure_latest
            else "- Structure latest settled `CLIP-S / LPIPS`: n/a"
        ),
        (
            f"- Structure latest `style - IDT`: transfer `{_style_delta_vs_idt(structure_latest.get('transfer_clip_style'), idt_refs.get('transfer_clip_style'))}`, "
            f"all-pairs `{_style_delta_vs_idt(structure_latest.get('all_pairs_clip_style'), idt_refs.get('all_pairs_clip_style'))}`"
            if structure_latest
            else "- Structure latest `style - IDT`: n/a"
        ),
        f"- Structure runtime observability: {_best_observability_brief(remote_structure.get('latest_summary'), remote_structure.get('latest_train_metrics'))}",
        f"- I2SB diagnostic preferred packet: `{i2sb.get('packet_id', 'n/a')}`",
        f"- I2SB config: {i2sb_cfg}" if i2sb_cfg else "- I2SB config: n/a",
        f"- I2SB note: {i2sb_note}" if i2sb_note else "- I2SB note: n/a",
        f"- I2SB read: {i2sb.get('current_read', 'n/a')}",
        f"- I2SB live state: `{i2sb_live_state}`",
        f"- I2SB runtime observability: {_best_observability_brief(remote_i2sb.get('latest_summary'), remote_i2sb.get('latest_train_metrics'))}",
        "",
        "## Contract Read",
        "- `true I2SB` is already implemented as exact-Brownian endpoint transport with `solver_i2sb`.",
        "- `true tokenizer` is already implemented as `pure_latent_spatial` with a null legacy tokenizer shell and structured runtime path.",
        "- The current formal lane remains on `velocity + pure_latent_spatial` because the exact-I2SB line has not returned to the documented `LPIPS < 0.40` band.",
        "",
        "## Remote Host Read",
        f"- SSH ok: `{health.get('ssh_ok', 'n/a')}`",
        f"- WSL exec ok: `{health.get('wsl_exec_ok', 'n/a')}`",
        f"- HCS failure: `{health.get('remote_wsl_hcs_failure', 'n/a')}`",
        f"- Hypervisor launch type: `{health.get('hypervisorlaunchtype', 'n/a')}`",
    ]
    if local_velocity_watchers:
        structure_stdout = str(local_structure_logs.get("out_log", "")).strip()
        lines.extend(
            [
                "",
                "## Local Watchers",
                f"- Active phase2 handoff watchers: `{len(local_velocity_watchers)}`",
                (
                    f"- Structure watcher stdout: {_path_link(structure_stdout, 'phase2_structure_reentry_watch.stdout.log')}"
                    if structure_stdout
                    else "- Structure watcher stdout: n/a"
                ),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a human-readable Phase-2 status note from the queue state snapshot.")
    parser.add_argument("--snapshot-json", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--date", default="")
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot_json).expanduser().resolve()
    output_path = Path(args.output_md).expanduser().resolve()
    report_date = str(args.date).strip() or "undated"

    snapshot = _load_json(snapshot_path)
    text = build_note(snapshot, report_date=report_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
