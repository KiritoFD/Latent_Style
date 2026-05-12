#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

def _fmt_float(x: Any, *, sig: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return "na"
    if v == 0:
        return "0"
    s = f"{v:.{sig}g}"
    s = s.replace("+0", "").replace("+", "")
    return s


def _safe_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _extract_epoch(summary: Dict[str, Any], summary_path: Path) -> Any:
    for part in summary_path.parts:
        m = re.fullmatch(r"epoch_(\d+)", part)
        if m:
            return int(m.group(1))
    ckpt = str(summary.get("checkpoint", ""))
    m = re.search(r"epoch_(\d+)", ckpt.replace("\\", "/"))
    if m:
        return int(m.group(1))
    return ""


def _extract_epoch_from_string(path_like: str) -> Any:
    if not path_like:
        return ""
    m = re.search(r"epoch_(\d+)", path_like.replace("\\", "/"))
    if m:
        return int(m.group(1))
    return ""


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_summary_path(raw: str, repo_root: Path) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p

    s = raw.replace("\\", "/")
    aliases = [
        ("I:/Github/Latent_Style", str(repo_root).replace("\\", "/")),
        ("G:/GitHub/Latent_Style", str(repo_root).replace("\\", "/")),
    ]
    for old, new in aliases:
        if s.lower().startswith(old.lower()):
            cand = Path(new + s[len(old):])
            if cand.exists():
                return cand

    cand = (repo_root / Path(raw)).resolve()
    if cand.exists():
        return cand
    return None


def _resolve_any_path(raw: str, repo_root: Path) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    s = raw.replace("\\", "/")
    aliases = [
        ("I:/Github/Latent_Style", str(repo_root).replace("\\", "/")),
        ("G:/GitHub/Latent_Style", str(repo_root).replace("\\", "/")),
    ]
    for old, new in aliases:
        if s.lower().startswith(old.lower()):
            cand = Path(new + s[len(old):])
            if cand.exists():
                return cand
    cand = (repo_root / Path(raw)).resolve()
    if cand.exists():
        return cand
    return None


def _build_experiment_id_from_config(config_path: Optional[Path], trial_number: int, params: Dict[str, Any]) -> str:
    base = f"trial_{trial_number:04d}"
    lr = params.get("lr")
    wswd = params.get("w_swd")
    wc = params.get("w_color")
    hf = params.get("hf_ratio")
    wid = params.get("w_idt")

    if config_path is not None and config_path.exists():
        try:
            cfg = _load_json(config_path)
            lr = _safe_get(cfg, "training", "learning_rate") if lr is None else lr
            wswd = _safe_get(cfg, "loss", "w_swd") if wswd is None else wswd
            wc = _safe_get(cfg, "loss", "w_color") if wc is None else wc
            hf = _safe_get(cfg, "loss", "swd_hf_weight_ratio") if hf is None else hf
            wid = _safe_get(cfg, "loss", "w_identity") if wid is None else wid
        except Exception:
            pass

    return "_".join(
        [
            base,
            f"lr{_fmt_float(lr)}",
            f"swd{_fmt_float(wswd)}",
            f"wc{_fmt_float(wc)}",
            f"hf{_fmt_float(hf)}",
            f"id{_fmt_float(wid)}",
        ]
    )


def _extract_row(summary: Dict[str, Any], summary_path: Path, experiment_id: str) -> Dict[str, Any]:
    transfer = _safe_get(summary, "analysis", "style_transfer_ability") or {}
    p2a = _safe_get(summary, "analysis", "photo_to_art_performance") or {}
    row = {
        "experiment_id": experiment_id,
        "epoch": _extract_epoch(summary, summary_path),
        "source_file": str(summary_path),
        "updated_at": str(summary.get("timestamp", "")),
        "transfer_clip_style": transfer.get("clip_style"),
        "transfer_clip_content": transfer.get("clip_content"),
        "transfer_content_lpips": transfer.get("content_lpips"),
        "transfer_fid": transfer.get("fid"),
        "transfer_art_fid": transfer.get("art_fid"),
        "transfer_classifier_acc": transfer.get("classifier_acc"),
        "photo_to_art_clip_style": p2a.get("clip_style"),
        "photo_to_art_clip_content": p2a.get("clip_content"),
        "photo_to_art_fid": p2a.get("fid"),
        "photo_to_art_art_fid": p2a.get("art_fid"),
        "photo_to_art_classifier_acc": p2a.get("classifier_acc"),
        "summary_path": str(summary_path),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser("Export Optuna study to opt.csv (config-based experiment_id)")
    ap.add_argument("--db", required=True, help="Path to .db file, e.g. style_transfer_hpo_e60.db")
    ap.add_argument("--study-name", default="", help="Study name; default first study in DB")
    ap.add_argument("--output", "-o", required=True, help="Output CSV path")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    studies = cur.execute("SELECT study_id, study_name FROM studies ORDER BY study_id").fetchall()
    if not studies:
        raise RuntimeError(f"No studies found in {db_path}")
    if args.study_name:
        matched = [r for r in studies if str(r["study_name"]) == str(args.study_name)]
        if not matched:
            names = ", ".join(str(r["study_name"]) for r in studies)
            raise RuntimeError(f"Study '{args.study_name}' not found. Available: {names}")
        study_id = int(matched[0]["study_id"])
        study_name = str(matched[0]["study_name"])
    else:
        study_id = int(studies[0]["study_id"])
        study_name = str(studies[0]["study_name"])

    repo_root = Path(__file__).resolve().parents[1]
    rows: List[Dict[str, Any]] = []
    missing_summary = 0

    trials = cur.execute(
        "SELECT trial_id, number, state FROM trials WHERE study_id = ? ORDER BY number",
        (study_id,),
    ).fetchall()
    for t in trials:
        trial_id = int(t["trial_id"])
        trial_number = int(t["number"])
        state = str(t["state"])
        if state != "COMPLETE":
            continue

        attr_rows = cur.execute(
            "SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?",
            (trial_id,),
        ).fetchall()
        attrs: Dict[str, Any] = {}
        for ar in attr_rows:
            k = str(ar["key"])
            vj = ar["value_json"]
            try:
                attrs[k] = json.loads(vj) if vj is not None else None
            except Exception:
                attrs[k] = vj

        param_rows = cur.execute(
            "SELECT param_name, param_value FROM trial_params WHERE trial_id = ?",
            (trial_id,),
        ).fetchall()
        params = {str(r["param_name"]): r["param_value"] for r in param_rows}

        summary_raw = str(attrs.get("summary_path", "") or "")
        summary_path = _resolve_summary_path(summary_raw, repo_root)

        exp_dir_raw = str(attrs.get("exp_dir", "") or "")
        exp_dir = _resolve_any_path(exp_dir_raw, repo_root) if exp_dir_raw else None
        cfg_candidates = []
        if exp_dir is not None:
            cfg_candidates.append((exp_dir / "config.json"))
            cfg_candidates.append((exp_dir.parent / "config.json"))
        cfg_path = next((p for p in cfg_candidates if p.exists()), None)

        exp_id = _build_experiment_id_from_config(cfg_path, trial_number, params)
        if summary_path is not None and summary_path.exists():
            try:
                summary = _load_json(summary_path)
                rows.append(_extract_row(summary, summary_path, exp_id))
                continue
            except Exception:
                pass

        missing_summary += 1
        # Fallback: derive row from DB attrs/params when summary.json is unavailable.
        rows.append(
            {
                "experiment_id": exp_id,
                "epoch": _extract_epoch_from_string(summary_raw),
                "source_file": summary_raw,
                "updated_at": "",
                "transfer_clip_style": attrs.get("clip_style"),
                "transfer_clip_content": "",
                "transfer_content_lpips": attrs.get("lpips"),
                "transfer_fid": "",
                "transfer_art_fid": "",
                "transfer_classifier_acc": "",
                "photo_to_art_clip_style": "",
                "photo_to_art_clip_content": "",
                "photo_to_art_fid": "",
                "photo_to_art_art_fid": "",
                "photo_to_art_classifier_acc": "",
                "summary_path": summary_raw,
            }
        )

    rows.sort(key=lambda r: (str(r.get("experiment_id", "")), int(r.get("epoch") or 0)))
    out_csv = Path(args.output).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        out_csv.write_text("", encoding="utf-8")
        print(f"[DONE] no rows -> {out_csv}")
        return

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] study={study_name} rows={len(rows)} missing_summary={missing_summary} -> {out_csv}")


if __name__ == "__main__":
    main()
