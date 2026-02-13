#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    src_dir: Path
    base_config: Path
    output_root: Path
    generated_dir: Path
    report_dir: Path
    manifest: Path


def _env_default(name: str, fallback: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else fallback


def _boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_get(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _get_cross_pair_clip(summary: dict[str, Any], src_name: str, tgt_name: str) -> Any:
    return _safe_get(summary, ["matrix_breakdown", src_name, tgt_name, "clip_style"], None)


def _resolve_paths(args: argparse.Namespace) -> Paths:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    base_config = Path(args.base_config).resolve()
    run_name = args.run_name.strip() if args.run_name else f"style_ablation_{dt.datetime.now():%Y%m%d_%H%M%S}"
    output_root = Path(args.output_root).resolve() if args.output_root else (repo_root / run_name).resolve()
    generated_dir = output_root / "configs"
    report_dir = output_root / "reports"
    manifest = output_root / "variants.tsv"
    return Paths(
        repo_root=repo_root,
        src_dir=src_dir,
        base_config=base_config,
        output_root=output_root,
        generated_dir=generated_dir,
        report_dir=report_dir,
        manifest=manifest,
    )


def _variant_registry(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    model_base = copy.deepcopy(base.get("model", {}))
    loss_base = copy.deepcopy(base.get("loss", {}))
    infer_base = copy.deepcopy(base.get("inference", {}))
    _ = infer_base  # keep for future extension

    variants: dict[str, dict[str, Any]] = {}

    def add_variant(name: str, category: str, patch: dict[str, Any], note: str) -> None:
        if name in variants:
            raise ValueError(f"Duplicate variant name: {name}")
        variants[name] = {"category": category, "patch": patch, "note": note}

    add_variant("baseline", "baseline", {}, "Base config")

    bool_injection_keys = [
        "use_decoder_spatial_inject",
        "use_delta_highpass_bias",
        "use_style_delta_gate",
        "use_decoder_adagn",
        "use_style_spatial_highpass",
        "normalize_style_spatial_maps",
        "use_output_style_affine",
        "use_style_spatial_blur",
    ]
    for k in bool_injection_keys:
        if k not in model_base:
            continue
        cur = _boolish(model_base.get(k))
        tgt = not cur
        suffix = "on" if tgt else "off"
        add_variant(
            f"inj_{k}_{suffix}",
            "injection_bool",
            {"model": {k: tgt}},
            f"Toggle {k} from {cur} to {tgt}",
        )

    numeric_injection_keys = [
        "style_ref_gain",
        "style_spatial_pre_gain_16",
        "style_spatial_dec_gain_32",
        "style_texture_gain",
        "style_delta_lowfreq_gain",
        "highpass_last_step_scale",
        "style_gate_floor",
    ]
    for k in numeric_injection_keys:
        if k not in model_base:
            continue
        base_v = float(model_base[k])
        if abs(base_v) > 1e-12:
            add_variant(f"inj_{k}_off", "injection_gain", {"model": {k: 0.0}}, f"Set {k}=0 from {base_v}")
        down = base_v * 0.5
        up = base_v * 1.5
        if any(tag in k for tag in ("floor", "scale", "gain")):
            down = _clamp01(down)
            up = _clamp01(up)
        if abs(down - base_v) > 1e-12:
            add_variant(f"inj_{k}_x0p5", "injection_gain", {"model": {k: down}}, f"Scale {k} to 0.5x ({base_v}->{down})")
        if abs(up - base_v) > 1e-12:
            add_variant(f"inj_{k}_x1p5", "injection_gain", {"model": {k: up}}, f"Scale {k} to 1.5x ({base_v}->{up})")

    if "style_texture_mode" in model_base:
        cur = str(model_base["style_texture_mode"]).lower().strip()
        alt = "style_only" if cur != "style_only" else "content_aware"
        add_variant(
            f"inj_style_texture_mode_{alt}",
            "injection_mode",
            {"model": {"style_texture_mode": alt}},
            f"Switch style_texture_mode {cur}->{alt}",
        )

    if "style_strength_step_curve" in model_base:
        cur = str(model_base["style_strength_step_curve"]).lower().strip()
        for alt in ("linear", "smoothstep", "sqrt"):
            if alt == cur:
                continue
            add_variant(
                f"inj_step_curve_{alt}",
                "injection_mode",
                {"model": {"style_strength_step_curve": alt}},
                f"Switch style_strength_step_curve {cur}->{alt}",
            )

    if "highpass_last_step_only" in model_base:
        cur = _boolish(model_base["highpass_last_step_only"])
        add_variant(
            f"inj_highpass_last_step_only_{'on' if not cur else 'off'}",
            "injection_mode",
            {"model": {"highpass_last_step_only": (not cur)}},
            f"Toggle highpass_last_step_only from {cur}",
        )

    inj_all_off_patch: dict[str, Any] = {"model": {}}
    for k in bool_injection_keys:
        if k in model_base:
            inj_all_off_patch["model"][k] = False
    for k in numeric_injection_keys:
        if k in model_base:
            inj_all_off_patch["model"][k] = 0.0
    if "style_texture_mode" in model_base:
        inj_all_off_patch["model"]["style_texture_mode"] = "style_only"
    if inj_all_off_patch["model"]:
        add_variant(
            "inj_all_style_paths_off",
            "injection_combo",
            inj_all_off_patch,
            "Disable most style injection branches",
        )

    inj_style_heavy: dict[str, Any] = {"model": {}}
    if "style_spatial_pre_gain_16" in model_base:
        inj_style_heavy["model"]["style_spatial_pre_gain_16"] = _clamp01(float(model_base["style_spatial_pre_gain_16"]) * 1.5)
    if "style_spatial_dec_gain_32" in model_base:
        inj_style_heavy["model"]["style_spatial_dec_gain_32"] = _clamp01(float(model_base["style_spatial_dec_gain_32"]) * 1.5)
    if "style_texture_gain" in model_base:
        inj_style_heavy["model"]["style_texture_gain"] = _clamp01(float(model_base["style_texture_gain"]) * 1.6)
    if "highpass_last_step_scale" in model_base:
        inj_style_heavy["model"]["highpass_last_step_scale"] = _clamp01(float(model_base["highpass_last_step_scale"]) * 1.5)
    if "use_delta_highpass_bias" in model_base:
        inj_style_heavy["model"]["use_delta_highpass_bias"] = True
    if inj_style_heavy["model"]:
        add_variant("inj_style_heavy", "injection_combo", inj_style_heavy, "Increase major style injection gains")

    loss_weight_keys = [
        "w_distill",
        "w_code",
        "w_struct",
        "w_edge",
        "w_cycle",
        "w_stroke_gram",
        "w_color_moment",
        "w_style_spatial_tv",
        "w_nce",
        "w_push",
        "w_delta_tv",
        "w_semigroup",
    ]
    probe_on_defaults = {
        "w_distill": 0.3,
        "w_code": 1.0,
        "w_struct": 0.1,
        "w_edge": 0.1,
        "w_cycle": 0.1,
        "w_stroke_gram": 40.0,
        "w_color_moment": 6.0,
        "w_style_spatial_tv": 0.002,
        "w_nce": 0.08,
        "w_push": 0.5,
        "w_delta_tv": 0.002,
        "w_semigroup": 0.08,
    }

    for k in loss_weight_keys:
        if k not in loss_base:
            continue
        base_v = float(loss_base[k])
        if abs(base_v) > 1e-12:
            add_variant(f"loss_{k}_off", "loss_single", {"loss": {k: 0.0}}, f"Disable {k} from {base_v}")
            add_variant(f"loss_{k}_x0p5", "loss_single", {"loss": {k: base_v * 0.5}}, f"Half {k}")
            add_variant(f"loss_{k}_x1p5", "loss_single", {"loss": {k: base_v * 1.5}}, f"Increase {k}")
        else:
            add_variant(f"loss_{k}_on", "loss_single", {"loss": {k: probe_on_defaults[k]}}, f"Probe-enable {k}")

    style_loss_keys = [k for k in ("w_stroke_gram", "w_color_moment", "w_style_spatial_tv") if k in loss_base]
    if style_loss_keys:
        add_variant(
            "loss_style_bundle_off",
            "loss_bundle",
            {"loss": {k: 0.0 for k in style_loss_keys}},
            "Disable all explicit style losses",
        )
        add_variant(
            "loss_style_bundle_up",
            "loss_bundle",
            {"loss": {k: float(loss_base[k]) * 1.5 if float(loss_base[k]) != 0.0 else probe_on_defaults.get(k, 1.0) for k in style_loss_keys}},
            "Increase all style losses",
        )

    if all(k in loss_base for k in ("w_struct", "w_edge", "w_nce")):
        add_variant(
            "loss_content_guard_relaxed",
            "loss_bundle",
            {"loss": {"w_struct": float(loss_base["w_struct"]) * 0.5, "w_edge": float(loss_base["w_edge"]) * 0.5, "w_nce": 0.0}},
            "Relax content-preserving terms",
        )
        add_variant(
            "loss_content_guard_strict",
            "loss_bundle",
            {"loss": {"w_struct": max(float(loss_base["w_struct"]), 0.1) * 1.5, "w_edge": max(float(loss_base["w_edge"]), 0.1) * 1.5, "w_nce": max(float(loss_base["w_nce"]), 0.08)}},
            "Strengthen content-preserving terms",
        )

    if "train_style_strength_min" in loss_base and "train_style_strength_max" in loss_base:
        lo = float(loss_base["train_style_strength_min"])
        hi = float(loss_base["train_style_strength_max"])
        add_variant(
            "sched_style_strength_fixed_1p0",
            "schedule",
            {"loss": {"train_style_strength_min": 1.0, "train_style_strength_max": 1.0}},
            "Fix style strength to 1.0",
        )
        add_variant(
            "sched_style_strength_wide",
            "schedule",
            {"loss": {"train_style_strength_min": _clamp01(min(lo, 0.85)), "train_style_strength_max": _clamp01(max(hi, 1.0))}},
            "Widen style strength range",
        )

    if "train_num_steps_min" in loss_base and "train_num_steps_max" in loss_base:
        smin = int(loss_base["train_num_steps_min"])
        smax = int(loss_base["train_num_steps_max"])
        add_variant(
            "sched_steps_1",
            "schedule",
            {"loss": {"train_num_steps_min": 1, "train_num_steps_max": 1}},
            "Force single-step train path",
        )
        add_variant(
            "sched_steps_plus1",
            "schedule",
            {"loss": {"train_num_steps_min": max(1, smin + 1), "train_num_steps_max": max(1, smax + 1)}},
            "Increase train integration steps by +1",
        )

    return variants


def build_variants(args: argparse.Namespace, paths: Paths) -> None:
    if not paths.base_config.exists():
        raise FileNotFoundError(f"Base config not found: {paths.base_config}")

    with open(paths.base_config, "r", encoding="utf-8") as f:
        base = json.load(f)

    variants = _variant_registry(base)
    if args.variants.strip():
        wanted = {x.strip() for x in args.variants.split(",") if x.strip()}
        unknown = sorted(wanted.difference(variants.keys()))
        if unknown:
            raise ValueError(f"Unknown variants in --variants: {unknown}")
        variants = {k: v for k, v in variants.items() if k in wanted}

    paths.generated_dir.mkdir(parents=True, exist_ok=True)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)

    lines: list[tuple[str, str, str, str, str]] = []
    for name, item in variants.items():
        cfg = copy.deepcopy(base)
        _deep_update(cfg, item["patch"])

        cfg.setdefault("training", {})
        cfg["training"]["num_epochs"] = int(args.epochs)
        cfg["training"]["save_interval"] = max(1, int(args.save_interval))
        cfg["training"]["full_eval_interval"] = max(0, int(args.full_eval_interval))
        cfg["training"]["full_eval_on_last_epoch"] = True
        cfg["training"]["resume_checkpoint"] = ""

        cfg.setdefault("inference", {})
        if "train_style_strength_max" in cfg.get("loss", {}) and "style_strength" in cfg.get("inference", {}):
            cfg["inference"]["style_strength"] = float(cfg["loss"]["train_style_strength_max"])

        run_dir = (paths.output_root / name).resolve()
        cfg.setdefault("checkpoint", {})
        cfg["checkpoint"]["save_dir"] = str(run_dir)

        out_cfg = (paths.generated_dir / f"{name}.json").resolve()
        with open(out_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        lines.append((name, item["category"], str(out_cfg), str(run_dir), item["note"]))

    with open(paths.manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["variant", "category", "config_path", "run_dir", "note"])
        writer.writerows(lines)

    print(f"Generated {len(lines)} ablation configs")
    print(f"Manifest: {paths.manifest}")


def train_variants(args: argparse.Namespace, paths: Paths) -> None:
    if not paths.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {paths.manifest}")

    with open(paths.manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    for r in rows:
        variant = r["variant"]
        category = r.get("category", "")
        config_path = r["config_path"]
        run_dir = r["run_dir"]
        note = r.get("note", "")

        print("\n============================================================")
        print(f"Variant : {variant} ({category})")
        print(f"Config  : {config_path}")
        print(f"Run dir : {run_dir}")
        print(f"Note    : {note}")
        print("============================================================")

        cmd = [args.python_bin, "run.py", "--config", config_path]
        proc = subprocess.run(cmd, cwd=str(paths.src_dir))
        if proc.returncode != 0:
            raise RuntimeError(f"Training failed for variant={variant} with code={proc.returncode}")


def report_variants(args: argparse.Namespace, paths: Paths) -> None:
    if not paths.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {paths.manifest}")

    target_epoch_dir = f"epoch_{int(args.epochs):04d}"
    rows_out: list[dict[str, Any]] = []

    with open(paths.manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    for r in rows:
        variant = r["variant"]
        category = r.get("category", "")
        note = r.get("note", "")
        run_dir = Path(r["run_dir"]).resolve()

        summary_path = run_dir / "full_eval" / target_epoch_dir / "summary.json"
        history_path = run_dir / "full_eval" / "summary_history.json"

        final: dict[str, Any] = {}
        best: dict[str, Any] = {}
        summary_exists = summary_path.exists()
        history_exists = history_path.exists()

        if summary_exists:
            try:
                with open(summary_path, "r", encoding="utf-8") as sf:
                    final = json.load(sf)
            except Exception:
                final = {}

        if history_exists:
            try:
                with open(history_path, "r", encoding="utf-8") as hf:
                    hist = json.load(hf)
                best = _safe_get(hist, ["best", "best_transfer_clip_style"], {}) or {}
            except Exception:
                best = {}

        rows_out.append(
            {
                "variant": variant,
                "category": category,
                "summary_exists": int(summary_exists),
                "history_exists": int(history_exists),
                "final_transfer_clip_style": _safe_get(final, ["analysis", "style_transfer_ability", "clip_style"], None),
                "best_transfer_clip_style": _safe_get(best, ["transfer_clip_style"], None),
                "best_epoch": _safe_get(best, ["epoch"], None),
                "final_photo_to_art_clip_style": _safe_get(final, ["analysis", "photo_to_art_performance", "clip_style"], None),
                "best_photo_to_art_clip_style": _safe_get(best, ["photo_to_art_clip_style"], None),
                "final_photo_to_hayao_clip_style": _get_cross_pair_clip(final, "photo", "Hayao"),
                "final_hayao_to_photo_clip_style": _get_cross_pair_clip(final, "Hayao", "photo"),
                "final_transfer_classifier_acc": _safe_get(final, ["analysis", "style_transfer_ability", "classifier_acc"], None),
                "best_transfer_classifier_acc": _safe_get(best, ["transfer_classifier_acc"], None),
                "final_classification_accuracy": _safe_get(final, ["classification_report", "accuracy"], None),
                "final_transfer_content_lpips": _safe_get(final, ["analysis", "style_transfer_ability", "content_lpips"], None),
                "summary_path": str(summary_path),
                "note": note,
            }
        )

    def sort_key(x: dict[str, Any]) -> tuple[float, float]:
        a = x.get("best_transfer_clip_style")
        b = x.get("final_transfer_clip_style")
        aa = -1.0 if a is None else float(a)
        bb = -1.0 if b is None else float(b)
        return (aa, bb)

    rows_out.sort(key=sort_key, reverse=True)

    paths.report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = paths.report_dir / "ablation_summary.csv"
    md_path = paths.report_dir / "ablation_summary.md"

    columns = [
        "variant",
        "category",
        "summary_exists",
        "history_exists",
        "final_transfer_clip_style",
        "best_transfer_clip_style",
        "best_epoch",
        "final_photo_to_art_clip_style",
        "best_photo_to_art_clip_style",
        "final_photo_to_hayao_clip_style",
        "final_hayao_to_photo_clip_style",
        "final_transfer_classifier_acc",
        "best_transfer_classifier_acc",
        "final_classification_accuracy",
        "final_transfer_content_lpips",
        "summary_path",
        "note",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows_out)

    def fmt(v: Any) -> str:
        if v is None:
            return "NA"
        if isinstance(v, (int, float)):
            return f"{float(v):.6f}"
        return str(v)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Style Ablation Summary\n\n")
        f.write("| variant | category | final_transfer | best_transfer | best_epoch | final_p2a | final_photo->Hayao | final_Hayao->photo | final_cls_acc | summary |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_out:
            f.write(
                f"| {r['variant']} | {r['category']} | {fmt(r['final_transfer_clip_style'])} | "
                f"{fmt(r['best_transfer_clip_style'])} | {fmt(r['best_epoch'])} | "
                f"{fmt(r['final_photo_to_art_clip_style'])} | {fmt(r['final_photo_to_hayao_clip_style'])} | "
                f"{fmt(r['final_hayao_to_photo_clip_style'])} | {fmt(r['final_classification_accuracy'])} | "
                f"{int(r['summary_exists'])} |\n"
            )

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    parser = argparse.ArgumentParser(description="Style ablation manager")
    parser.add_argument("--mode", default=_env_default("MODE", "all"), choices=["all", "build", "train", "report"])
    parser.add_argument("--python-bin", default=_env_default("PYTHON_BIN", sys.executable))
    parser.add_argument("--base-config", default=_env_default("BASE_CONFIG", str(src_dir / "config.json")))
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", ""))
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", ""))
    parser.add_argument("--epochs", type=int, default=int(_env_default("EPOCHS", "60")))
    parser.add_argument("--save-interval", type=int, default=int(_env_default("SAVE_INTERVAL", "20")))
    parser.add_argument("--full-eval-interval", type=int, default=int(_env_default("FULL_EVAL_INTERVAL", "20")))
    parser.add_argument("--variants", default=_env_default("VARIANTS", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _resolve_paths(args)

    print(f"Repo root      : {paths.repo_root}")
    print(f"Base config    : {paths.base_config}")
    print(f"Output root    : {paths.output_root}")
    print(f"Mode           : {args.mode}")
    print(f"Epochs         : {args.epochs}")
    print(f"Variant filter : {args.variants if args.variants else '<all>'}")

    if args.mode == "build":
        build_variants(args, paths)
    elif args.mode == "train":
        train_variants(args, paths)
    elif args.mode == "report":
        report_variants(args, paths)
    else:
        build_variants(args, paths)
        train_variants(args, paths)
        report_variants(args, paths)

    print("Done.")


if __name__ == "__main__":
    main()

