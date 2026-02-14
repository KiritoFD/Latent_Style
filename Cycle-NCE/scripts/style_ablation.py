#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


LEGACY_LOSS_KEYS = {
    "w_gram",
    "w_moment",
    "w_idt",
    "nce_warmup_epochs",
    "nce_ramp_epochs",
    "cycle_warmup_epochs",
    "cycle_ramp_epochs",
    "struct_warmup_epochs",
    "struct_ramp_epochs",
    "edge_warmup_epochs",
    "edge_ramp_epochs",
    "idt_warmup_epochs",
    "idt_ramp_epochs",
    "teacher_interval_steps",
    "stroke_interval_steps",
    "semigroup_interval_steps",
}


@dataclass(frozen=True)
class VariantDef:
    name: str
    category: str
    note: str
    overrides: dict[str, Any]


@dataclass
class VariantRunResult:
    variant: VariantDef
    config_path: Path
    run_dir: Path
    status: str
    return_code: int | None = None
    error: str = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _safe_get(data: dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _set_by_dotted_path(payload: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur: dict[str, Any] = payload
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _get_by_dotted_path(payload: dict[str, Any], dotted: str, default: Any) -> Any:
    cur: Any = payload
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _normalize_loss_aliases(loss_cfg: dict[str, Any]) -> None:
    if "w_stroke_gram" not in loss_cfg and "w_gram" in loss_cfg:
        loss_cfg["w_stroke_gram"] = _as_float(loss_cfg.get("w_gram"), 0.0)
    if "w_color_moment" not in loss_cfg and "w_moment" in loss_cfg:
        loss_cfg["w_color_moment"] = _as_float(loss_cfg.get("w_moment"), 0.0)


def _drop_legacy_loss_keys(loss_cfg: dict[str, Any]) -> list[str]:
    removed: list[str] = []
    for key in sorted(LEGACY_LOSS_KEYS):
        if key in loss_cfg:
            removed.append(key)
            loss_cfg.pop(key, None)
    return removed


def _sanitize_train_ranges(cfg: dict[str, Any]) -> None:
    loss_cfg = cfg.setdefault("loss", {})
    smin = _clip(_as_float(loss_cfg.get("train_style_strength_min", 1.0), 1.0), 0.0, 1.0)
    smax = _clip(_as_float(loss_cfg.get("train_style_strength_max", smin), smin), 0.0, 1.0)
    if smax < smin:
        smin, smax = smax, smin
    loss_cfg["train_style_strength_min"] = smin
    loss_cfg["train_style_strength_max"] = smax

    nmin = max(1, _as_int(loss_cfg.get("train_num_steps_min", 1), 1))
    nmax = max(1, _as_int(loss_cfg.get("train_num_steps_max", nmin), nmin))
    if nmax < nmin:
        nmin, nmax = nmax, nmin
    loss_cfg["train_num_steps_min"] = nmin
    loss_cfg["train_num_steps_max"] = nmax

    hmin = _as_float(loss_cfg.get("train_step_size_min", 1.0), 1.0)
    hmax = _as_float(loss_cfg.get("train_step_size_max", hmin), hmin)
    if hmin <= 0.0:
        hmin = 1.0
    if hmax <= 0.0:
        hmax = hmin
    if hmax < hmin:
        hmin, hmax = hmax, hmin
    loss_cfg["train_step_size_min"] = hmin
    loss_cfg["train_step_size_max"] = hmax


def _short_run_base(base_cfg: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    cfg = copy.deepcopy(base_cfg)

    cfg.setdefault("model", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("training", {})
    cfg.setdefault("inference", {})
    cfg.setdefault("checkpoint", {})

    loss_cfg = cfg["loss"]
    _normalize_loss_aliases(loss_cfg)
    removed_legacy = _drop_legacy_loss_keys(loss_cfg)

    train_cfg = cfg["training"]
    infer_cfg = cfg["inference"]

    epochs = max(1, int(args.epochs))
    train_cfg["num_epochs"] = epochs
    train_cfg["resume_checkpoint"] = ""
    train_cfg["full_eval_on_last_epoch"] = True
    train_cfg["snapshot_source"] = bool(args.snapshot_source)

    if args.disable_compile:
        train_cfg["use_compile"] = False

    if args.save_interval > 0:
        save_interval = args.save_interval
    else:
        save_interval = max(1, epochs // 2)
    train_cfg["save_interval"] = min(save_interval, epochs)

    if args.eval_interval > 0:
        eval_interval = args.eval_interval
    else:
        eval_interval = max(1, epochs // 5)
    train_cfg["full_eval_interval"] = min(eval_interval, epochs)

    if _as_int(train_cfg.get("log_interval", 0), 0) <= 0:
        train_cfg["log_interval"] = max(1, int(args.default_log_interval))

    train_cfg.setdefault("scheduler", "cosine")
    lr = _as_float(train_cfg.get("learning_rate", 2e-4), 2e-4)
    min_lr = _as_float(train_cfg.get("min_learning_rate", lr * 0.05), lr * 0.05)
    if min_lr <= 0.0 or min_lr >= lr:
        min_lr = max(1e-6, lr * 0.05)
    train_cfg["min_learning_rate"] = min_lr

    train_cfg.setdefault("full_eval_num_steps", _as_int(infer_cfg.get("num_steps", 1), 1))
    train_cfg.setdefault("full_eval_step_size", _as_float(infer_cfg.get("step_size", 1.0), 1.0))
    if "style_strength" in infer_cfg and infer_cfg.get("style_strength") is not None:
        train_cfg.setdefault("full_eval_style_strength", _clip(_as_float(infer_cfg.get("style_strength"), 1.0), 0.0, 1.0))

    loss_cfg["train_num_steps_min"] = max(1, min(_as_int(loss_cfg.get("train_num_steps_min", 1), 1), 3))
    loss_cfg["train_num_steps_max"] = max(
        loss_cfg["train_num_steps_min"],
        min(_as_int(loss_cfg.get("train_num_steps_max", loss_cfg["train_num_steps_min"]), loss_cfg["train_num_steps_min"]), 3),
    )

    _sanitize_train_ranges(cfg)
    return cfg, removed_legacy


def _scale(cfg: dict[str, Any], dotted: str, factor: float, default: float, *, lo: float = 0.0, hi: float = 1e9) -> float:
    base = _as_float(_get_by_dotted_path(cfg, dotted, default), default)
    return _clip(base * factor, lo, hi)


def _shift(cfg: dict[str, Any], dotted: str, delta: float, default: float, *, lo: float = 0.0, hi: float = 1e9) -> float:
    base = _as_float(_get_by_dotted_path(cfg, dotted, default), default)
    return _clip(base + delta, lo, hi)

def _build_variants(short_base: dict[str, Any], mode: str) -> list[VariantDef]:
    variants_all: list[VariantDef] = [
        VariantDef(
            name="baseline_50e",
            category="baseline",
            note="50-epoch short-run baseline (legacy warmup keys removed).",
            overrides={},
        ),
        VariantDef(
            name="inj_no_decoder_spatial",
            category="inj_single",
            note="Disable decoder spatial inject path.",
            overrides={"model.use_decoder_spatial_inject": False},
        ),
        VariantDef(
            name="inj_no_texture_gain",
            category="inj_single",
            note="Disable texture residual head contribution.",
            overrides={"model.style_texture_gain": 0.0},
        ),
        VariantDef(
            name="inj_no_delta_highpass",
            category="inj_single",
            note="Disable delta high-frequency bias.",
            overrides={"model.use_delta_highpass_bias": False},
        ),
        VariantDef(
            name="inj_no_delta_gate",
            category="inj_single",
            note="Disable style delta gate.",
            overrides={"model.use_style_delta_gate": False},
        ),
        VariantDef(
            name="inj_no_decoder_adagn",
            category="inj_single",
            note="Disable decoder AdaGN, fallback to GroupNorm.",
            overrides={"model.use_decoder_adagn": False},
        ),
        VariantDef(
            name="inj_no_output_affine",
            category="inj_single",
            note="Disable output style affine.",
            overrides={"model.use_output_style_affine": False},
        ),
        VariantDef(
            name="inj_spatial_highpass_on",
            category="inj_single",
            note="Enable high-pass style spatial maps.",
            overrides={"model.use_style_spatial_highpass": True},
        ),
        VariantDef(
            name="inj_spatial_map_norm_off",
            category="inj_single",
            note="Disable style spatial map normalization.",
            overrides={"model.normalize_style_spatial_maps": False},
        ),
        VariantDef(
            name="inj_spatial_blur_off",
            category="inj_single",
            note="Disable style spatial blur before injection.",
            overrides={"model.use_style_spatial_blur": False},
        ),
        VariantDef(
            name="loss_no_distill",
            category="loss_single",
            note="Disable distillation term.",
            overrides={"loss.w_distill": 0.0},
        ),
        VariantDef(
            name="loss_no_code",
            category="loss_single",
            note="Disable code-closure term.",
            overrides={"loss.w_code": 0.0},
        ),
        VariantDef(
            name="loss_no_stroke_gram",
            category="loss_single",
            note="Disable stroke gram style statistics.",
            overrides={"loss.w_stroke_gram": 0.0},
        ),
        VariantDef(
            name="loss_no_color_moment",
            category="loss_single",
            note="Disable color moment style statistics.",
            overrides={"loss.w_color_moment": 0.0},
        ),
        VariantDef(
            name="loss_no_style_spatial_tv",
            category="loss_single",
            note="Disable style spatial TV regularization.",
            overrides={"loss.w_style_spatial_tv": 0.0},
        ),
        VariantDef(
            name="loss_no_struct_edge",
            category="loss_single",
            note="Disable struct and edge guards.",
            overrides={"loss.w_struct": 0.0, "loss.w_edge": 0.0},
        ),
        VariantDef(
            name="loss_no_cycle",
            category="loss_single",
            note="Disable cycle consistency.",
            overrides={"loss.w_cycle": 0.0},
        ),
        VariantDef(
            name="loss_no_nce",
            category="loss_single",
            note="Disable NCE content term.",
            overrides={"loss.w_nce": 0.0},
        ),
        VariantDef(
            name="loss_no_semigroup",
            category="loss_single",
            note="Disable semigroup consistency.",
            overrides={"loss.w_semigroup": 0.0},
        ),
        VariantDef(
            name="loss_no_push",
            category="loss_single",
            note="Disable style push-away term.",
            overrides={"loss.w_push": 0.0},
        ),
        VariantDef(
            name="dyn_style_strength_jitter",
            category="dynamics",
            note="Widen train-time style strength range.",
            overrides={
                "loss.train_style_strength_min": 0.70,
                "loss.train_style_strength_max": 1.00,
                "inference.style_strength": 0.95,
            },
        ),
        VariantDef(
            name="dyn_num_steps_1_to_3",
            category="dynamics",
            note="Use variable training steps in [1,3].",
            overrides={
                "loss.train_num_steps_min": 1,
                "loss.train_num_steps_max": 3,
                "loss.train_step_schedule": "late",
            },
        ),
        VariantDef(
            name="dyn_single_step_only",
            category="dynamics",
            note="Force single-step training and inference.",
            overrides={
                "loss.train_num_steps_min": 1,
                "loss.train_num_steps_max": 1,
                "loss.train_step_size_min": 1.0,
                "loss.train_step_size_max": 1.0,
                "inference.num_steps": 1,
                "training.full_eval_num_steps": 1,
            },
        ),
        VariantDef(
            name="dyn_flat_step_schedule",
            category="dynamics",
            note="Use flat step schedule for train/infer.",
            overrides={
                "loss.train_step_schedule": "flat",
                "model.step_schedule_default": "flat",
                "inference.step_schedule": "flat",
                "training.full_eval_step_schedule": "flat",
            },
        ),
        VariantDef(
            name="bundle_style_up",
            category="bundle",
            note="Increase style injection + style losses for 50-epoch run.",
            overrides={
                "model.style_spatial_pre_gain_16": _scale(short_base, "model.style_spatial_pre_gain_16", 1.20, 0.35, lo=0.0, hi=2.0),
                "model.style_spatial_dec_gain_32": _scale(short_base, "model.style_spatial_dec_gain_32", 1.25, 0.18, lo=0.0, hi=2.0),
                "model.style_texture_gain": _scale(short_base, "model.style_texture_gain", 1.25, 0.12, lo=0.0, hi=2.5),
                "model.highpass_last_step_scale": _clip(_as_float(_get_by_dotted_path(short_base, "model.highpass_last_step_scale", 0.2), 0.2) + 0.05, 0.0, 1.0),
                "loss.w_distill": _scale(short_base, "loss.w_distill", 1.20, 0.2, lo=0.0, hi=1000.0),
                "loss.w_code": _scale(short_base, "loss.w_code", 1.20, 4.0, lo=0.0, hi=1000.0),
                "loss.w_stroke_gram": _scale(short_base, "loss.w_stroke_gram", 1.25, 30.0, lo=0.0, hi=2000.0),
                "loss.w_color_moment": _scale(short_base, "loss.w_color_moment", 1.25, 2.0, lo=0.0, hi=200.0),
                "loss.train_style_strength_min": 0.85,
                "loss.train_style_strength_max": 1.00,
                "inference.style_strength": 1.00,
            },
        ),
        VariantDef(
            name="bundle_style_down",
            category="bundle",
            note="Reduce style injection + style losses.",
            overrides={
                "model.style_spatial_pre_gain_16": _scale(short_base, "model.style_spatial_pre_gain_16", 0.70, 0.35, lo=0.0, hi=2.0),
                "model.style_spatial_dec_gain_32": _scale(short_base, "model.style_spatial_dec_gain_32", 0.70, 0.18, lo=0.0, hi=2.0),
                "model.style_texture_gain": _scale(short_base, "model.style_texture_gain", 0.70, 0.12, lo=0.0, hi=2.5),
                "loss.w_distill": _scale(short_base, "loss.w_distill", 0.70, 0.2, lo=0.0, hi=1000.0),
                "loss.w_code": _scale(short_base, "loss.w_code", 0.70, 4.0, lo=0.0, hi=1000.0),
                "loss.w_stroke_gram": _scale(short_base, "loss.w_stroke_gram", 0.70, 30.0, lo=0.0, hi=2000.0),
                "loss.w_color_moment": _scale(short_base, "loss.w_color_moment", 0.70, 2.0, lo=0.0, hi=200.0),
                "loss.train_style_strength_min": 0.60,
                "loss.train_style_strength_max": 0.90,
                "inference.style_strength": 0.85,
            },
        ),
        VariantDef(
            name="bundle_content_guard_strict",
            category="bundle",
            note="Increase structure-preservation losses.",
            overrides={
                "loss.w_struct": _scale(short_base, "loss.w_struct", 1.35, 0.1, lo=0.0, hi=1000.0),
                "loss.w_edge": _scale(short_base, "loss.w_edge", 1.35, 0.1, lo=0.0, hi=1000.0),
                "loss.w_cycle": _scale(short_base, "loss.w_cycle", 1.25, 0.1, lo=0.0, hi=1000.0),
                "loss.w_nce": _scale(short_base, "loss.w_nce", 1.25, 0.1, lo=0.0, hi=1000.0),
                "loss.struct_lowpass_strength": _shift(short_base, "loss.struct_lowpass_strength", 0.10, 0.35, lo=0.0, hi=1.0),
                "loss.cycle_lowpass_strength": _shift(short_base, "loss.cycle_lowpass_strength", 0.05, 0.05, lo=0.0, hi=1.0),
            },
        ),
        VariantDef(
            name="bundle_content_guard_relaxed",
            category="bundle",
            note="Reduce structure-preservation losses for style-first bias.",
            overrides={
                "loss.w_struct": _scale(short_base, "loss.w_struct", 0.65, 0.1, lo=0.0, hi=1000.0),
                "loss.w_edge": _scale(short_base, "loss.w_edge", 0.65, 0.1, lo=0.0, hi=1000.0),
                "loss.w_cycle": _scale(short_base, "loss.w_cycle", 0.65, 0.1, lo=0.0, hi=1000.0),
                "loss.w_nce": _scale(short_base, "loss.w_nce", 0.65, 0.1, lo=0.0, hi=1000.0),
                "loss.struct_lowpass_strength": _shift(short_base, "loss.struct_lowpass_strength", -0.10, 0.35, lo=0.0, hi=1.0),
                "loss.cycle_lowpass_strength": _shift(short_base, "loss.cycle_lowpass_strength", -0.03, 0.05, lo=0.0, hi=1.0),
            },
        ),
        VariantDef(
            name="bundle_style_losses_off",
            category="bundle",
            note="Turn off all style-specific loss terms.",
            overrides={
                "loss.w_distill": 0.0,
                "loss.w_code": 0.0,
                "loss.w_stroke_gram": 0.0,
                "loss.w_color_moment": 0.0,
                "loss.w_push": 0.0,
                "loss.w_style_spatial_tv": 0.0,
            },
        ),
        VariantDef(
            name="bundle_all_style_paths_off",
            category="bundle",
            note="Ablate main style-injection paths + style losses.",
            overrides={
                "model.use_decoder_spatial_inject": False,
                "model.style_texture_gain": 0.0,
                "model.use_style_delta_gate": False,
                "model.use_output_style_affine": False,
                "model.use_delta_highpass_bias": False,
                "loss.w_distill": 0.0,
                "loss.w_code": 0.0,
                "loss.w_stroke_gram": 0.0,
                "loss.w_color_moment": 0.0,
                "loss.w_push": 0.0,
                "loss.w_style_spatial_tv": 0.0,
            },
        ),
        VariantDef(
            name="bundle_style_injection_heavy",
            category="bundle",
            note="Strong style injection with extra smoothing regularization.",
            overrides={
                "model.use_style_spatial_highpass": True,
                "model.style_spatial_pre_gain_16": _scale(short_base, "model.style_spatial_pre_gain_16", 1.35, 0.35, lo=0.0, hi=3.0),
                "model.style_spatial_dec_gain_32": _scale(short_base, "model.style_spatial_dec_gain_32", 1.35, 0.18, lo=0.0, hi=3.0),
                "model.style_texture_gain": _scale(short_base, "model.style_texture_gain", 1.35, 0.12, lo=0.0, hi=3.0),
                "model.highpass_last_step_scale": _clip(_as_float(_get_by_dotted_path(short_base, "model.highpass_last_step_scale", 0.2), 0.2) + 0.10, 0.0, 1.0),
                "loss.w_delta_tv": _scale(short_base, "loss.w_delta_tv", 1.20, 0.002, lo=0.0, hi=10.0),
                "loss.w_style_spatial_tv": _scale(short_base, "loss.w_style_spatial_tv", 1.20, 0.001, lo=0.0, hi=10.0),
            },
        ),
    ]

    if mode == "all":
        return variants_all

    quick_keep = {
        "baseline_50e",
        "inj_no_decoder_spatial",
        "inj_no_texture_gain",
        "inj_no_delta_highpass",
        "inj_no_delta_gate",
        "loss_no_distill",
        "loss_no_code",
        "loss_no_stroke_gram",
        "loss_no_color_moment",
        "dyn_style_strength_jitter",
        "bundle_style_up",
        "bundle_content_guard_relaxed",
        "bundle_all_style_paths_off",
    }
    return [v for v in variants_all if v.name in quick_keep]


def _apply_variant(base_cfg: dict[str, Any], variant: VariantDef) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for dotted, value in variant.overrides.items():
        _set_by_dotted_path(cfg, dotted, value)
    _normalize_loss_aliases(cfg.setdefault("loss", {}))
    _drop_legacy_loss_keys(cfg["loss"])
    _sanitize_train_ranges(cfg)
    return cfg


def _write_variants_tsv(path: Path, rows: list[VariantRunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["variant", "category", "config_path", "run_dir", "status", "note"])
        for row in rows:
            writer.writerow(
                [
                    row.variant.name,
                    row.variant.category,
                    str(row.config_path.resolve()),
                    str(row.run_dir.resolve()),
                    row.status,
                    row.variant.note,
                ]
            )


def _parse_epoch_name(name: str) -> int | None:
    if name.startswith("epoch_"):
        tail = name.split("_", 1)[1]
        if tail.isdigit():
            return int(tail)
        return None
    if name.isdigit():
        return int(name)
    return None


def _latest_summary(run_dir: Path) -> tuple[int | None, Path | None, dict[str, Any] | None]:
    full_eval = run_dir / "full_eval"
    if not full_eval.is_dir():
        return None, None, None

    latest_ep: int | None = None
    latest_path: Path | None = None
    for p in full_eval.glob("*/summary.json"):
        ep = _parse_epoch_name(p.parent.name)
        if ep is None:
            continue
        if latest_ep is None or ep > latest_ep:
            latest_ep = ep
            latest_path = p
    if latest_path is None:
        return None, None, None
    try:
        return latest_ep, latest_path, _load_json(latest_path)
    except Exception:
        return latest_ep, latest_path, None


def _read_history_best(run_dir: Path) -> dict[str, Any]:
    out = {
        "best_transfer_clip_style": None,
        "best_transfer_epoch": None,
        "best_photo_to_art_clip_style": None,
        "best_photo_to_art_epoch": None,
    }
    history_path = run_dir / "full_eval" / "summary_history.json"
    if not history_path.exists():
        return out
    try:
        history = _load_json(history_path)
    except Exception:
        return out
    if not isinstance(history, dict):
        return out

    best_transfer = _safe_get(history, ["best", "best_transfer_clip_style"], {})
    if isinstance(best_transfer, dict):
        out["best_transfer_clip_style"] = best_transfer.get("transfer_clip_style")
        out["best_transfer_epoch"] = best_transfer.get("epoch")

    best_p2a = _safe_get(history, ["best", "best_photo_to_art_clip_style"], {})
    if isinstance(best_p2a, dict):
        out["best_photo_to_art_clip_style"] = best_p2a.get("photo_to_art_clip_style")
        out["best_photo_to_art_epoch"] = best_p2a.get("epoch")

    return out

def _read_last_training_metrics(run_dir: Path) -> dict[str, Any]:
    logs = run_dir / "logs"
    out = {"last_train_epoch": None, "last_train_loss": None, "last_train_lr": None}
    if not logs.is_dir():
        return out
    csvs = sorted(logs.glob("training_*.csv"))
    if not csvs:
        return out
    last_csv = csvs[-1]
    last_row: dict[str, str] | None = None
    try:
        with last_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                last_row = row
    except Exception:
        return out
    if last_row is None:
        return out
    try:
        out["last_train_epoch"] = int(float(last_row.get("epoch", "")))
    except Exception:
        out["last_train_epoch"] = None
    try:
        out["last_train_loss"] = float(last_row.get("loss", ""))
    except Exception:
        out["last_train_loss"] = None
    try:
        out["last_train_lr"] = float(last_row.get("lr", ""))
    except Exception:
        out["last_train_lr"] = None
    return out


def _as_float_or_none(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if v != v:
        return None
    return v


def _collect_summary_rows(results: list[VariantRunResult], epochs: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        latest_ep, latest_path, latest_summary = _latest_summary(r.run_dir)
        history_best = _read_history_best(r.run_dir)
        last_train = _read_last_training_metrics(r.run_dir)

        transfer_clip = None
        transfer_cls = None
        transfer_lpips = None
        p2a_clip = None
        p2a_cls = None
        if isinstance(latest_summary, dict):
            transfer_clip = _as_float_or_none(_safe_get(latest_summary, ["analysis", "style_transfer_ability", "clip_style"]))
            transfer_cls = _as_float_or_none(_safe_get(latest_summary, ["analysis", "style_transfer_ability", "classifier_acc"]))
            transfer_lpips = _as_float_or_none(_safe_get(latest_summary, ["analysis", "style_transfer_ability", "content_lpips"]))
            p2a_clip = _as_float_or_none(_safe_get(latest_summary, ["analysis", "photo_to_art_performance", "clip_style"]))
            p2a_cls = _as_float_or_none(_safe_get(latest_summary, ["analysis", "photo_to_art_performance", "classifier_acc"]))

        finished_ckpt = r.run_dir / f"epoch_{int(epochs):04d}.pt"
        completed = finished_ckpt.exists()

        rows.append(
            {
                "variant": r.variant.name,
                "category": r.variant.category,
                "status": r.status,
                "completed_target_epoch": completed,
                "target_epoch": int(epochs),
                "latest_eval_epoch": latest_ep,
                "latest_transfer_clip_style": transfer_clip,
                "latest_transfer_classifier_acc": transfer_cls,
                "latest_transfer_content_lpips": transfer_lpips,
                "latest_photo_to_art_clip_style": p2a_clip,
                "latest_photo_to_art_classifier_acc": p2a_cls,
                "best_transfer_clip_style": _as_float_or_none(history_best.get("best_transfer_clip_style")),
                "best_transfer_epoch": history_best.get("best_transfer_epoch"),
                "best_photo_to_art_clip_style": _as_float_or_none(history_best.get("best_photo_to_art_clip_style")),
                "best_photo_to_art_epoch": history_best.get("best_photo_to_art_epoch"),
                "last_train_epoch": last_train.get("last_train_epoch"),
                "last_train_loss": last_train.get("last_train_loss"),
                "last_train_lr": last_train.get("last_train_lr"),
                "config_path": str(r.config_path.resolve()),
                "run_dir": str(r.run_dir.resolve()),
                "latest_summary_path": "" if latest_path is None else str(latest_path.resolve()),
                "note": r.variant.note,
                "error": r.error,
                "return_code": r.return_code,
            }
        )
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "variant",
        "category",
        "status",
        "completed_target_epoch",
        "target_epoch",
        "latest_eval_epoch",
        "latest_transfer_clip_style",
        "latest_transfer_classifier_acc",
        "latest_transfer_content_lpips",
        "latest_photo_to_art_clip_style",
        "latest_photo_to_art_classifier_acc",
        "best_transfer_clip_style",
        "best_transfer_epoch",
        "best_photo_to_art_clip_style",
        "best_photo_to_art_epoch",
        "last_train_epoch",
        "last_train_loss",
        "last_train_lr",
        "config_path",
        "run_dir",
        "latest_summary_path",
        "note",
        "error",
        "return_code",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_num(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _write_summary_md(path: Path, rows: list[dict[str, Any]], *, mode: str, epochs: int, removed_legacy: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def sort_key(item: dict[str, Any]) -> tuple[int, float]:
        best = item.get("best_transfer_clip_style")
        latest = item.get("latest_transfer_clip_style")
        if best is not None:
            return (0, -float(best))
        if latest is not None:
            return (1, -float(latest))
        return (2, 0.0)

    ordered = sorted(rows, key=sort_key)
    lines: list[str] = []
    lines.append("# Style Ablation Summary")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Mode: `{mode}`")
    lines.append(f"- Target epochs per run: `{epochs}`")
    if removed_legacy:
        lines.append(f"- Removed legacy loss keys: `{', '.join(removed_legacy)}`")
    lines.append("")
    lines.append("| Variant | Category | Status | Latest transfer | Best transfer | Best epoch | Latest p2a | Last train loss | Note |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for row in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("variant", "")),
                    str(row.get("category", "")),
                    str(row.get("status", "")),
                    _fmt_num(row.get("latest_transfer_clip_style"), digits=4),
                    _fmt_num(row.get("best_transfer_clip_style"), digits=4),
                    _fmt_num(row.get("best_transfer_epoch"), digits=0),
                    _fmt_num(row.get("latest_photo_to_art_clip_style"), digits=4),
                    _fmt_num(row.get("last_train_loss"), digits=4),
                    str(row.get("note", "")),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_variant(
    result: VariantRunResult,
    *,
    python_exe: str,
    train_entry: Path,
    train_cwd: Path,
    run_args: list[str],
    skip_existing: bool,
    keep_going: bool,
    log_dir: Path,
    target_epoch: int,
) -> VariantRunResult:
    if skip_existing and (result.run_dir / f"epoch_{target_epoch:04d}.pt").exists():
        result.status = "skipped_existing"
        return result

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{result.variant.name}.log"
    cmd = [python_exe, str(train_entry), "--config", str(result.config_path)] + run_args
    result.status = "running"

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"CWD: {train_cwd}\n")
        logf.write("Command:\n")
        logf.write(" ".join(shlex.quote(part) for part in cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=str(train_cwd), stdout=logf, stderr=subprocess.STDOUT)
    result.return_code = int(proc.returncode)
    if proc.returncode == 0:
        result.status = "ok"
        return result

    result.status = "failed"
    result.error = f"run failed, see {log_path}"
    if not keep_going:
        raise RuntimeError(result.error)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and run 50-epoch style/loss ablations.")
    parser.add_argument("--base-config", type=str, default="src/config.json", help="Base config path.")
    parser.add_argument("--output-root", type=str, default="", help="Output directory root. Default: experiments-cycle/ablation50_<timestamp>.")
    parser.add_argument("--mode", type=str, choices=["quick", "all"], default="all", help="Variant set size.")
    parser.add_argument("--epochs", type=int, default=50, help="Target epochs per ablation run.")
    parser.add_argument("--save-interval", type=int, default=25, help="Checkpoint interval. <=0 uses auto.")
    parser.add_argument("--eval-interval", type=int, default=10, help="Full-eval interval. <=0 uses auto.")
    parser.add_argument("--default-log-interval", type=int, default=10, help="Fallback log interval if base config has <=0.")
    parser.add_argument("--snapshot-source", action="store_true", help="Enable training.snapshot_source for each run.")
    parser.add_argument("--disable-compile", action="store_true", help="Force training.use_compile=false for all variants.")
    parser.add_argument("--run", action="store_true", help="Actually run training after generating configs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have target epoch checkpoint.")
    parser.add_argument("--keep-going", action="store_true", help="Continue even if one variant fails.")
    parser.add_argument("--max-runs", type=int, default=0, help="Run only first N variants (>0).")
    parser.add_argument("--python-exe", type=str, default=sys.executable, help="Python executable for launching training.")
    parser.add_argument("--train-entry", type=str, default="src/run.py", help="Training entry script path.")
    parser.add_argument(
        "--train-cwd",
        type=str,
        default="",
        help="Working directory for training subprocess. Default: directory of --train-entry.",
    )
    parser.add_argument(
        "--run-args",
        type=str,
        default="",
        help="Extra args forwarded to training command, e.g. \"--resume xxx\".",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()

    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = (repo_root / base_config_path).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    train_entry = Path(args.train_entry)
    if not train_entry.is_absolute():
        train_entry = (repo_root / train_entry).resolve()
    if not train_entry.exists():
        raise FileNotFoundError(f"Training entry not found: {train_entry}")
    if args.train_cwd:
        train_cwd = Path(args.train_cwd)
        if not train_cwd.is_absolute():
            train_cwd = (repo_root / train_cwd).resolve()
    else:
        train_cwd = train_entry.parent
    if not train_cwd.exists() or not train_cwd.is_dir():
        raise FileNotFoundError(f"Training cwd not found or not a directory: {train_cwd}")

    if args.output_root:
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = (repo_root / output_root).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = (repo_root / "experiments-cycle" / f"ablation50_{stamp}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_json(base_config_path)
    short_base, removed_legacy = _short_run_base(base_cfg, args)
    variants = _build_variants(short_base, args.mode)

    if args.max_runs > 0:
        variants = variants[: int(args.max_runs)]

    configs_dir = output_root / "configs"
    logs_dir = output_root / "runner_logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[VariantRunResult] = []
    for variant in variants:
        run_dir = output_root / variant.name
        cfg = _apply_variant(short_base, variant)
        cfg["checkpoint"]["save_dir"] = str(run_dir.resolve())

        cfg_path = configs_dir / f"{variant.name}.json"
        _write_json(cfg_path, cfg)
        run_results.append(
            VariantRunResult(
                variant=variant,
                config_path=cfg_path,
                run_dir=run_dir,
                status="generated",
            )
        )

    if args.run:
        forwarded_args = shlex.split(args.run_args, posix=False) if args.run_args else []
        for i, result in enumerate(run_results, start=1):
            print(f"[{i}/{len(run_results)}] {result.variant.name}")
            updated = _run_variant(
                result,
                python_exe=args.python_exe,
                train_entry=train_entry,
                train_cwd=train_cwd,
                run_args=forwarded_args,
                skip_existing=bool(args.skip_existing),
                keep_going=bool(args.keep_going),
                log_dir=logs_dir,
                target_epoch=int(args.epochs),
            )
            result.status = updated.status
            result.return_code = updated.return_code
            result.error = updated.error

    variants_tsv_path = output_root / "variants.tsv"
    _write_variants_tsv(variants_tsv_path, run_results)

    summary_rows = _collect_summary_rows(run_results, epochs=int(args.epochs))
    summary_csv_path = output_root / "ablation_summary.csv"
    summary_md_path = output_root / "ablation_summary.md"
    _write_summary_csv(summary_csv_path, summary_rows)
    _write_summary_md(
        summary_md_path,
        summary_rows,
        mode=args.mode,
        epochs=int(args.epochs),
        removed_legacy=removed_legacy,
    )

    print(f"Output root: {output_root}")
    print(f"Variants: {len(run_results)}")
    print(f"TSV: {variants_tsv_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary MD: {summary_md_path}")


if __name__ == "__main__":
    main()
