"""Build curve CSV + manifest + control_delta upsert from style_overdrive eval summaries."""
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "docs" / "experiments" / "phase2_fiber_bundle"
EVAL_ROOT = BASE / "eval" / "style_overdrive_k070_e3"
CURVES = BASE / "curves"
CONTROL_PATH = BASE / "control_delta.csv"

VARIANTS = [
    ("s110", "s110", 110, 1.10, 0.0, "none", "pure_style_overdrive", "style_strength_max=1.10 eval-only from k070 epoch_0003"),
    ("s120", "s120", 120, 1.20, 0.0, "none", "lpips_target_positive", "style_strength_max=1.20 eval-only from k070 epoch_0003"),
    ("s135", "s135", 135, 1.35, 0.0, "none", "balanced_positive", "style_strength_max=1.35 eval-only from k070 epoch_0003"),
    ("s160", "s160", 160, 1.60, 0.0, "none", "style_overdrive_frontier", "style_strength_max=1.60 eval-only from k070 epoch_0003"),
    ("s135_lataff045", "s135_lataff045", 13545, 1.35, 0.45, "style_latent_affine", "combo_style_candidate", "style_strength_max=1.35 plus latent affine strength=0.45"),
    ("s160_lataff045", "s160_lataff045", 16045, 1.60, 0.45, "style_latent_affine", "style_ceiling_lpips_cost", "style_strength_max=1.60 plus latent affine strength=0.45"),
]
PARENT = {
    "transfer_clip": 0.6718203524251779,
    "transfer_lpips": 0.3146181108066667,
    "full_clip": 0.7032335336605707,
    "full_lpips": 0.3125496446986667,
}
IDT = {"transfer": 0.6399208252628644, "full": 0.6801226128737131}


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return f"{float(x):.12g}"


def _load_metrics(name: str) -> dict:
    d = EVAL_ROOT / name
    s = json.loads((d / "summary.json").read_text(encoding="utf-8-sig"))
    a = s["analysis"]
    rt = s.get("runtime_observability", {})
    transfer_rt = (rt.get("style_transfer_ability") or {}) if isinstance(rt, dict) else {}
    full_rt = (rt.get("all_pairs_overview") or {}) if isinstance(rt, dict) else {}
    settings = s.get("settings", {})
    timings = s.get("timings_sec", {})
    return {
        "checkpoint": s.get("checkpoint", ""),
        "timestamp": s.get("timestamp", ""),
        "transfer_clip_style": a["style_transfer_ability"]["clip_style"],
        "transfer_content_lpips": a["style_transfer_ability"]["content_lpips"],
        "all_pairs_clip_style": a["all_pairs_overview"]["clip_style"],
        "all_pairs_content_lpips": a["all_pairs_overview"]["content_lpips"],
        "identity_clip_style": a["identity_reconstruction"]["clip_style"],
        "identity_content_lpips": a["identity_reconstruction"]["content_lpips"],
        "wall_total_seconds": timings.get("wall_total"),
        "eval_total_sec": timings.get("eval_total"),
        "generation_sec": timings.get("lancet_generation"),
        "vae_decode_sec": timings.get("vae_decode"),
        "latent_postprocess_mode": settings.get("latent_postprocess_mode", "none"),
        "latent_postprocess_strength": settings.get("latent_postprocess_strength", 0.0),
        "runtime_style_strength_effective": transfer_rt.get("style_strength_effective"),
        "runtime_style_strength_max": transfer_rt.get("style_strength_max"),
        "runtime_integration_horizon": transfer_rt.get("integration_horizon"),
        "runtime_style_step_scale": transfer_rt.get("style_step_scale"),
        "runtime_latent_style_affine_strength": transfer_rt.get("latent_style_affine_strength"),
        "runtime_full_integration_horizon": full_rt.get("integration_horizon"),
        "summary_path": str(d / "summary.json"),
        "metrics_path": str(d / "metrics.csv"),
    }


def _build_rows():
    rows = []
    for name, step, epoch_int, strength, lataff, mode, decision, note in VARIANTS:
        m = _load_metrics(name)
        rows.append({
            "epoch": step,
            "epoch_int": epoch_int,
            "checkpoint": m["checkpoint"],
            "timestamp": m["timestamp"],
            "style_strength": strength,
            "style_strength_max": strength,
            "latent_postprocess_mode": mode,
            "latent_postprocess_strength": lataff,
            "transfer_clip_style": m["transfer_clip_style"],
            "transfer_content_lpips": m["transfer_content_lpips"],
            "all_pairs_clip_style": m["all_pairs_clip_style"],
            "all_pairs_content_lpips": m["all_pairs_content_lpips"],
            "identity_clip_style": m["identity_clip_style"],
            "identity_content_lpips": m["identity_content_lpips"],
            "wall_total_seconds": m["wall_total_seconds"],
            "eval_total_sec": m["eval_total_sec"],
            "generation_sec": m["generation_sec"],
            "vae_decode_sec": m["vae_decode_sec"],
            "runtime_style_strength_effective": m["runtime_style_strength_effective"],
            "runtime_style_strength_max": m["runtime_style_strength_max"],
            "runtime_integration_horizon": m["runtime_integration_horizon"],
            "runtime_style_step_scale": m["runtime_style_step_scale"],
            "runtime_latent_style_affine_strength": m["runtime_latent_style_affine_strength"],
            "summary_path": m["summary_path"],
            "metrics_path": m["metrics_path"],
            "decision": decision,
            "note": note,
        })
    return rows


def _write_curves(rows):
    CURVES.mkdir(parents=True, exist_ok=True)
    curve_fields = list(rows[0].keys())
    all_curve = CURVES / "style_overdrive_all_k070_e3_eval_only_curve.csv"
    with all_curve.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=curve_fields)
        w.writeheader()
        w.writerows([{k: _fmt(v) for k, v in r.items()} for r in rows])
    subsets = [
        ("style_overdrive_k070_e3_eval_only_curve.csv",
         [r for r in rows if r["latent_postprocess_mode"] == "none"]),
        ("style_overdrive_lataff045_k070_e3_eval_only_curve.csv",
         [r for r in rows if r["latent_postprocess_mode"] != "none"]),
    ]
    for out_name, subset in subsets:
        with (CURVES / out_name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=curve_fields)
            w.writeheader()
            w.writerows([{k: _fmt(v) for k, v in r.items()} for r in subset])
    print("wrote", all_curve)


def _write_manifest(rows):
    mfields = [
        "family_id", "switches", "parent_checkpoint", "remote_output_root",
        "local_eval_root", "vram_band", "best_ckpt",
        "transfer_clip_style", "transfer_content_lpips",
        "all_pairs_clip_style", "all_pairs_content_lpips",
        "style_minus_idt_transfer", "style_minus_idt_full",
        "convergence_reason", "decision_status", "summary_path", "metrics_path",
    ]
    mrows = []
    for r in rows:
        switches = (
            f"model.style_strength_max={r['style_strength_max']}; "
            f"eval.style_strength={r['style_strength']}; "
            f"full_eval.latent_postprocess_mode={r['latent_postprocess_mode']}; "
            f"full_eval.latent_postprocess_strength={r['latent_postprocess_strength']}"
        )
        mrows.append({
            "family_id": "style_overdrive_k070_e3_" + r["epoch"],
            "switches": switches,
            "parent_checkpoint": r["checkpoint"],
            "remote_output_root": "/mnt/i/Github/Latent_Style/exp/inmortal-exp/phase2_style_overdrive_k070_e3",
            "local_eval_root": str(EVAL_ROOT / r["epoch"]),
            "vram_band": "eval-only health around 2.5-2.8 GiB; formal cap <11.0 GiB",
            "best_ckpt": r["checkpoint"],
            "transfer_clip_style": r["transfer_clip_style"],
            "transfer_content_lpips": r["transfer_content_lpips"],
            "all_pairs_clip_style": r["all_pairs_clip_style"],
            "all_pairs_content_lpips": r["all_pairs_content_lpips"],
            "style_minus_idt_transfer": r["transfer_clip_style"] - IDT["transfer"],
            "style_minus_idt_full": r["all_pairs_clip_style"] - IDT["full"],
            "convergence_reason": "eval-only matched sweep completed; no training convergence claim",
            "decision_status": r["decision"],
            "summary_path": r["summary_path"],
            "metrics_path": r["metrics_path"],
        })
    path = BASE / "style_overdrive_eval_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mfields)
        w.writeheader()
        w.writerows([{k: _fmt(v) for k, v in r.items()} for r in mrows])
    print("wrote", path)


def _upsert_control_delta(rows):
    dfnames = [
        "family_id", "scope", "parent_clip_style", "parent_content_lpips",
        "control_clip_style", "control_content_lpips",
        "candidate_clip_style", "candidate_content_lpips",
        "parent_style_minus_idt", "control_style_minus_idt", "candidate_style_minus_idt",
        "candidate_minus_control_clip_style", "candidate_minus_control_content_lpips",
        "decision", "notes",
    ]
    existing = []
    if CONTROL_PATH.exists():
        with CONTROL_PATH.open("r", encoding="utf-8-sig", newline="") as f:
            existing = list(csv.DictReader(f))
    by_key = {(r.get("family_id", ""), r.get("scope", "")): r for r in existing}
    counts = {}
    for r in rows:
        for scope in ("transfer", "full"):
            pclip = PARENT["transfer_clip" if scope == "transfer" else "full_clip"]
            plp = PARENT["transfer_lpips" if scope == "transfer" else "full_lpips"]
            cclip = r["transfer_clip_style" if scope == "transfer" else "all_pairs_clip_style"]
            clp = r["transfer_content_lpips" if scope == "transfer" else "all_pairs_content_lpips"]
            fid = "style_overdrive_k070_e3_" + r["epoch"]
            note = (
                f"{r['note']}; "
                f"style_strength overdrive confirms the previous hard <=1 clamp was suppressing style response"
            )
            by_key[(fid, scope)] = {
                "family_id": fid,
                "scope": scope,
                "parent_clip_style": _fmt(pclip),
                "parent_content_lpips": _fmt(plp),
                "control_clip_style": _fmt(pclip),
                "control_content_lpips": _fmt(plp),
                "candidate_clip_style": _fmt(cclip),
                "candidate_content_lpips": _fmt(clp),
                "parent_style_minus_idt": _fmt(pclip - IDT[scope]),
                "control_style_minus_idt": _fmt(pclip - IDT[scope]),
                "candidate_style_minus_idt": _fmt(cclip - IDT[scope]),
                "candidate_minus_control_clip_style": _fmt(cclip - pclip),
                "candidate_minus_control_content_lpips": _fmt(clp - plp),
                "decision": r["decision"],
                "notes": note,
            }
            counts.setdefault(scope, 0)
            counts[scope] += 1
    ordered = sorted(by_key.values(), key=lambda r: (r.get("family_id", ""), r.get("scope", "")))
    with CONTROL_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dfnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows([{k: rr.get(k, "") for k in dfnames} for rr in ordered])
    print("upserted control_delta rows transfer={} full={} total={}".format(counts.get("transfer", 0), counts.get("full", 0), len(ordered)))


def main():
    rows = _build_rows()
    _write_curves(rows)
    _write_manifest(rows)
    _upsert_control_delta(rows)


if __name__ == "__main__":
    main()
