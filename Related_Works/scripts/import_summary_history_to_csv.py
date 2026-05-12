#!/usr/bin/env python3
"""
Import summary/history JSON data to CSV with per-epoch metrics.
Supports incremental updates with deduplication.

Usage:
    python import_summary_history_to_csv.py --input <file_or_dir> --output output.csv
    python import_summary_history_to_csv.py --input . --output metrics.csv
"""

import argparse
import csv
import json
import re
import sqlite3
from collections import OrderedDict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _is_tokenized_path_parts(parts: List[str]) -> bool:
    for part in parts:
        p = str(part).lower()
        if "tokenized" in p:
            return True
    return False


def _with_tokenized_suffix(exp_id: str, is_tokenized: bool) -> str:
    if not exp_id:
        return exp_id
    if not is_tokenized:
        return exp_id
    if exp_id.endswith("_tokenized"):
        return exp_id
    return f"{exp_id}_tokenized"


def _extract_distill_epochs_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    normalized = str(text).replace("\\", "/")
    m = re.search(r"distill_epochs[_-]?(\d+)", normalized)
    if not m:
        return None
    try:
        return str(int(m.group(1)))
    except Exception:
        return m.group(1)


def _with_distill_suffix(exp_id: str, distill_epochs: Optional[str]) -> str:
    if not exp_id or not distill_epochs:
        return exp_id
    suffix = f"_distill_epochs{distill_epochs}"
    if exp_id.endswith(suffix):
        return exp_id
    return f"{exp_id}{suffix}"


def _split_experiment_suffixes(exp_id: str) -> Tuple[str, Optional[str], bool]:
    tokenized = exp_id.endswith("_tokenized")
    base = exp_id[:-10] if tokenized else exp_id
    m = re.search(r"_distill_epochs(\d+)$", base)
    distill_epochs = m.group(1) if m else None
    if m:
        base = base[:m.start()]
    return base, distill_epochs, tokenized


def _split_path_parts(path_text: str) -> List[str]:
    normalized = str(path_text).replace("\\", "/")
    return [p for p in normalized.split("/") if p not in {"", "."}]


def _normalize_experiment_candidate(parts: List[str], idx: int) -> str:
    """
    Normalize experiment id candidate around `parts[idx]`:
    - default: current segment
    - if current segment is a generic container like 'experiment(s)',
      fallback to one-level-up segment as experiment id.
    """
    if idx < 0 or idx >= len(parts):
        return ""
    candidate = str(parts[idx]).strip()
    if not candidate:
        return ""
    lc = candidate.lower()
    if lc in {"experiment", "experiments"} and idx - 1 >= 0:
        upper = str(parts[idx - 1]).strip()
        if upper and upper not in {".", ".."}:
            return upper
    return candidate


def _read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file with error handling."""
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Error reading {path}: {e}")
        return {}


def _extract_experiment_id_from_summary_path(summary_path: str) -> str:
   
    if not summary_path:
        return 'unknown'
    
    # Normalize path separators
    normalized = summary_path.replace('\\', '/')
    parts = _split_path_parts(normalized)
    tokenized = _is_tokenized_path_parts(parts)
    distill_epochs = _extract_distill_epochs_from_text(normalized)
    
    # Find the directory before "full_eval", "eval", "epoch_*", or "summary.json"
    for i, part in enumerate(parts):
        if part in {'full_eval', 'eval'} or part.startswith('epoch_'):
            if i > 0:
                exp_name = _normalize_experiment_candidate(parts, i - 1)
                if exp_name and exp_name not in {'.', '..'}:
                    exp_name = _with_distill_suffix(exp_name, distill_epochs)
                    return _with_tokenized_suffix(exp_name, tokenized)
    
    # Fallback: return first non-trivial directory name
    for part in parts:
        if part and part not in {'.', '..', 'summary.json'}:
            part = _with_distill_suffix(part, distill_epochs)
            return _with_tokenized_suffix(part, tokenized)
    
    return 'unknown'


def _extract_rounds_data(json_data: Dict[str, Any], source_path: Path) -> List[Dict[str, Any]]:
    """
    Extract per-epoch records from summary_history.json.
    
    Expected structure:
    {
        "rounds": [
            {
                "epoch": 20,
                "summary_path": "..\\decoder-H-MSCTM-no_clamp_mult-tv-2\\full_eval\\epoch_0020\\summary.json",
                "transfer_clip_style": 0.655,
                "transfer_content_lpips": 0.416,
                ...
            },
            ...
        ]
    }
    """
    records = []
    
    if 'rounds' not in json_data or not isinstance(json_data['rounds'], list):
        return records
    
    for round_data in json_data['rounds']:
        if not isinstance(round_data, dict):
            continue
            
        # Extract experiment ID from summary_path in the round data
        summary_path = round_data.get('summary_path', '')
        experiment_id = _extract_experiment_id_from_summary_path(summary_path)
        
        record = {
            'source_file': str(source_path),
            'experiment_id': experiment_id,
            'updated_at': json_data.get('updated_at', ''),
        }
        
        # Copy all fields from round_data (will override if keys conflict)
        record.update(round_data)
        
        # Ensure epoch is present
        if 'epoch' not in record:
            continue
            
        records.append(record)
    
    return records


def _extract_epoch_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        m = re.fullmatch(r"epoch_(\d+)", part)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def _extract_epoch_from_checkpoint_string(s: str) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"epoch_(\d+)", s.replace("\\", "/"))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_experiment_id_from_file_path(path: Path) -> str:
    parts = list(path.parts)
    tokenized = _is_tokenized_path_parts([str(p) for p in parts])
    distill_epochs = _extract_distill_epochs_from_text(str(path))
    for i, part in enumerate(parts):
        part_l = str(part).lower()
        if part_l == 'full_eval' and i > 0:
            candidate = _normalize_experiment_candidate([str(p) for p in parts], i - 1)
            if candidate:
                candidate = _with_distill_suffix(candidate, distill_epochs)
                return _with_tokenized_suffix(candidate, tokenized)
        # Support variants like "full_eval_tokenized", "full_eval_xxx".
        if part_l.startswith('full_eval') and i > 0:
            candidate = _normalize_experiment_candidate([str(p) for p in parts], i - 1)
            if candidate:
                candidate = _with_distill_suffix(candidate, distill_epochs)
                return _with_tokenized_suffix(candidate, tokenized)
    fallback = _with_distill_suffix(path.parent.name or 'unknown', distill_epochs)
    return _with_tokenized_suffix(fallback, tokenized)


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


def _build_trial_label(
    trial_number: int,
    params: Dict[str, Any],
    user_attrs: Dict[str, Any],
    config_path: Optional[Path] = None,
) -> str:
    lr = params.get("lr_v2", params.get("lr"))
    wswd = params.get("w_swd")
    wc = params.get("w_color")
    hf = params.get("hf_ratio")
    wid = params.get("w_idt", user_attrs.get("w_idt"))

    if config_path is not None and config_path.exists():
        cfg = _read_json(config_path)
        lr = _safe_get(cfg, "training", "learning_rate") if lr is None else lr
        wswd = _safe_get(cfg, "loss", "w_swd") if wswd is None else wswd
        wc = _safe_get(cfg, "loss", "w_color") if wc is None else wc
        hf = _safe_get(cfg, "loss", "swd_hf_weight_ratio") if hf is None else hf
        wid = _safe_get(cfg, "loss", "w_identity") if wid is None else wid

    return "_".join(
        [
            f"trial_{trial_number:04d}",
            f"lr{_fmt_float(lr)}",
            f"swd{_fmt_float(wswd)}",
            f"wc{_fmt_float(wc)}",
            f"hf{_fmt_float(hf)}",
            f"id{_fmt_float(wid)}",
        ]
    )


def _build_optuna_trial_label_map(source: Path) -> Dict[str, str]:
    root = source if source.is_dir() else source.parent
    db_files = sorted(root.glob("*.db")) + sorted(root.glob("**/*.db"))
    mapping: Dict[str, str] = {}

    for db_path in db_files:
        try:
            con = sqlite3.connect(str(db_path))
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            trials = cur.execute("SELECT trial_id, number FROM trials ORDER BY trial_id").fetchall()
            for t in trials:
                trial_id = int(t["trial_id"])
                trial_number = int(t["number"])
                short_id = f"trial_{trial_number:04d}"

                p_rows = cur.execute(
                    "SELECT param_name, param_value FROM trial_params WHERE trial_id = ?",
                    (trial_id,),
                ).fetchall()
                params = {str(r["param_name"]): r["param_value"] for r in p_rows}

                a_rows = cur.execute(
                    "SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?",
                    (trial_id,),
                ).fetchall()
                user_attrs: Dict[str, Any] = {}
                for ar in a_rows:
                    k = str(ar["key"])
                    vj = ar["value_json"]
                    try:
                        user_attrs[k] = json.loads(vj) if vj is not None else None
                    except Exception:
                        user_attrs[k] = vj

                config_path: Optional[Path] = None
                exp_dir = user_attrs.get("exp_dir")
                if isinstance(exp_dir, str) and exp_dir:
                    cand = Path(exp_dir) / "config.json"
                    if cand.exists():
                        config_path = cand

                mapping[short_id] = _build_trial_label(trial_number, params, user_attrs, config_path)
            con.close()
        except Exception as e:
            print(f"[WARN] Failed reading db {db_path}: {e}")

    return mapping


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _extract_clip_content_from_matrix(json_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Fallback extraction for clip_content when analysis sections do not expose it.

    Returns:
      (transfer_clip_content, photo_to_art_clip_content)
    """
    matrix = json_data.get('matrix_breakdown')
    if not isinstance(matrix, dict):
        return None, None

    transfer_vals: List[float] = []
    photo_to_art_vals: List[float] = []

    for src_name, src_row in matrix.items():
        if not isinstance(src_row, dict):
            continue
        for tgt_name, cell in src_row.items():
            if not isinstance(cell, dict):
                continue
            value = cell.get('clip_content')
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue

            # Transfer proxy: all off-diagonal pairs.
            if str(src_name) != str(tgt_name):
                transfer_vals.append(float(value))

            # Photo->Art proxy: photo row to non-photo targets.
            if str(src_name) == 'photo' and str(tgt_name) != 'photo':
                photo_to_art_vals.append(float(value))

    return _safe_mean(transfer_vals), _safe_mean(photo_to_art_vals)


def _extract_single_summary_data(json_data: Dict[str, Any], source_path: Path) -> List[Dict[str, Any]]:
    """
    Extract one row from full_eval/epoch_xxxx/summary.json.
    """
    analysis = json_data.get('analysis')
    if not isinstance(analysis, dict):
        return []

    all_pairs = analysis.get('all_pairs_overview')
    transfer = analysis.get('style_transfer_ability')
    photo_to_art = analysis.get('photo_to_art_performance')
    if not isinstance(all_pairs, dict) or not isinstance(transfer, dict) or not isinstance(photo_to_art, dict):
        return []

    matrix_transfer_clip_content, matrix_photo_to_art_clip_content = _extract_clip_content_from_matrix(json_data)
    transfer_clip_content = transfer.get('clip_content')
    if transfer_clip_content is None:
        transfer_clip_content = matrix_transfer_clip_content

    photo_to_art_clip_content = photo_to_art.get('clip_content')
    if photo_to_art_clip_content is None:
        photo_to_art_clip_content = matrix_photo_to_art_clip_content

    epoch = _extract_epoch_from_path(source_path)
    if epoch is None:
        epoch = _extract_epoch_from_checkpoint_string(str(json_data.get('checkpoint', '')))
    record = {
        'source_file': str(source_path),
        'experiment_id': _extract_experiment_id_from_file_path(source_path),
        'updated_at': json_data.get('timestamp', ''),
        'summary_path': str(source_path),
        'epoch': int(epoch) if epoch is not None else '',
        'all_clip_style': all_pairs.get('clip_style'),
        'all_clip_content': all_pairs.get('clip_content'),
        'all_content_lpips': all_pairs.get('content_lpips'),
        'all_fid': all_pairs.get('fid'),
        'all_art_fid': all_pairs.get('art_fid'),
        'all_classifier_acc': all_pairs.get('classifier_acc'),
        'transfer_clip_style': transfer.get('clip_style'),
        'transfer_clip_content': transfer_clip_content,
        'transfer_content_lpips': transfer.get('content_lpips'),
        'transfer_fid': transfer.get('fid'),
        'transfer_art_fid': transfer.get('art_fid'),
        'transfer_classifier_acc': transfer.get('classifier_acc'),
        'photo_to_art_clip_style': photo_to_art.get('clip_style'),
        'photo_to_art_clip_content': photo_to_art_clip_content,
        'photo_to_art_fid': photo_to_art.get('fid'),
        'photo_to_art_art_fid': photo_to_art.get('art_fid'),
        'photo_to_art_classifier_acc': photo_to_art.get('classifier_acc'),
    }
    return [record]


def _get_all_json_files(source: Path, recursive: bool = False, summary_only: bool = False) -> List[Path]:
    """Find all supported json files (summary_history*.json and summary*.json)."""
    def _is_supported_json(p: Path) -> bool:
        if p.suffix.lower() != '.json':
            return False
        name = p.name.lower()
        if summary_only:
            return name == 'summary.json'
        return name.startswith('summary_history') or name.startswith('summary')

    if source.is_file():
        if _is_supported_json(source):
            return [source]
        return []
    
    if not source.is_dir():
        return []
    
    pattern = '**/*.json' if recursive else '*.json'
    return [p for p in source.glob(pattern) if _is_supported_json(p)]


def _flatten_record(record: Dict[str, Any]) -> Dict[str, str]:
    """Flatten nested structures to strings for CSV."""
    flat = OrderedDict()
    
    def _flatten_helper(obj: Any, prefix: str = ''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}_{k}" if prefix else k
                _flatten_helper(v, new_key)
        elif isinstance(obj, (list, tuple)):
            flat[prefix] = str(obj)
        else:
            flat[prefix] = str(obj) if obj is not None else ''
    
    _flatten_helper(record)
    return flat


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip() != ""


def _pick_first_non_empty(row: Dict[str, str], keys: List[str]) -> str:
    for key in keys:
        if _is_non_empty(row.get(key, "")):
            return str(row.get(key, ""))
    return ""


def _normalize_bom_keys_in_row(row: Dict[str, str]) -> Dict[str, str]:
    """
    Normalize potential BOM-prefixed header names (e.g. '\\ufeffexperiment_id').
    Keep the first non-empty value when keys collapse to same normalized name.
    """
    normalized: Dict[str, str] = {}
    for raw_key, raw_val in row.items():
        key = str(raw_key).lstrip("\ufeff")
        val = str(raw_val) if raw_val is not None else ""
        if key not in normalized:
            normalized[key] = val
            continue
        # Prefer non-empty value over empty.
        if (not _is_non_empty(normalized[key])) and _is_non_empty(val):
            normalized[key] = val
    return normalized


def _normalize_metric_columns(row: Dict[str, str]) -> Dict[str, str]:
    """
    Unified metrics layer:
    1) If all_* is missing, fallback to transfer_* for legacy runs.
    2) Emit canonical columns clip_style / clip_content / content_lpips.
    """
    legacy_backfill = [
        ("all_clip_style", "transfer_clip_style"),
        ("all_clip_content", "transfer_clip_content"),
        ("all_content_lpips", "transfer_content_lpips"),
        ("all_fid", "transfer_fid"),
        ("all_art_fid", "transfer_art_fid"),
        ("all_classifier_acc", "transfer_classifier_acc"),
    ]
    for dst, src in legacy_backfill:
        if (not _is_non_empty(row.get(dst, ""))) and _is_non_empty(row.get(src, "")):
            row[dst] = str(row.get(src, ""))

    row["clip_style"] = _pick_first_non_empty(
        row,
        ["clip_style", "all_clip_style", "transfer_clip_style", "photo_to_art_clip_style"],
    )
    row["clip_content"] = _pick_first_non_empty(
        row,
        ["clip_content", "all_clip_content", "transfer_clip_content", "photo_to_art_clip_content"],
    )
    row["content_lpips"] = _pick_first_non_empty(
        row,
        ["content_lpips", "all_content_lpips", "transfer_content_lpips"],
    )
    return row


def _normalize_row_schema(row: Dict[str, str]) -> Dict[str, str]:
    row = _normalize_bom_keys_in_row(row)
    row = _normalize_metric_columns(row)
    return row


def _get_dedup_key(record: Dict[str, str]) -> str:
    """Generate key for deduplication."""
    exp_id = record.get('experiment_id', 'unknown')
    epoch = record.get('epoch', '0')
    source = record.get('source_file', '')
    return f"{exp_id}|{epoch}|{source}"


def _normalize_experiment_id_in_row(row: Dict[str, str]) -> Dict[str, str]:
    exp_id = str(row.get("experiment_id", "") or "")
    summary_path = str(row.get("summary_path", "") or "")
    source_file = str(row.get("source_file", "") or "")
    probe = summary_path if summary_path else source_file
    if not probe:
        return row
    parts = _split_path_parts(probe)
    distill_epochs = _extract_distill_epochs_from_text(probe)
    exp_id = _with_distill_suffix(exp_id, distill_epochs)
    if _is_tokenized_path_parts(parts):
        row["experiment_id"] = _with_tokenized_suffix(exp_id, True)
    else:
        row["experiment_id"] = exp_id
    return row


def _apply_trial_label_map(row: Dict[str, str], label_map: Dict[str, str]) -> Dict[str, str]:
    exp_id = str(row.get("experiment_id", "") or "")
    base, distill_epochs, tokenized = _split_experiment_suffixes(exp_id)
    mapped = label_map.get(base)
    if mapped:
        mapped = _with_distill_suffix(mapped, distill_epochs)
        row["experiment_id"] = _with_tokenized_suffix(mapped, tokenized)
    return row


def _read_existing_csv(csv_path: Path) -> Tuple[List[Dict[str, str]], Set[str]]:
    """Read existing CSV and extract dedup keys."""
    if not csv_path.exists():
        return [], set()
    
    rows = []
    keys = set()
    
    try:
        with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return [], set()
            
            for row in reader:
                row = _normalize_row_schema(row)
                row = _normalize_experiment_id_in_row(row)
                rows.append(row)
                key = _get_dedup_key(row)
                keys.add(key)
        
        return rows, keys
    except Exception as e:
        print(f"[WARN] Error reading existing CSV: {e}")
        return [], set()


def _merge_and_deduplicate(
    existing: List[Dict[str, str]],
    existing_keys: Set[str],
    new_records: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], int]:
    """Merge records with deduplication, return merged list and count of added."""
    added = 0
    
    for record in new_records:
        key = _get_dedup_key(record)
        if key not in existing_keys:
            existing.append(record)
            existing_keys.add(key)
            added += 1
    
    return existing, added


def _normalize_numeric_string(value: Any) -> Optional[str]:
    """
    Normalize numeric string for exact-value comparison.
    Returns None for non-numeric or empty values.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None

    lowered = text.lower()
    if lowered in {"nan", "none", "null", "inf", "+inf", "-inf"}:
        return None

    try:
        num = Decimal(text)
    except (InvalidOperation, ValueError):
        return None

    if num == 0:
        return "0"
    return str(num.normalize())


def _build_numeric_signature(record: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    """
    Build a deterministic signature from all numeric columns in a row.
    Two rows with identical signatures have fully matching numeric values.
    """
    ignore_fields = {"experiment_id", "source_file", "updated_at", "summary_path"}
    signature: List[Tuple[str, str]] = []

    for key in sorted(record.keys()):
        if key in ignore_fields:
            continue
        normalized = _normalize_numeric_string(record.get(key, ""))
        if normalized is not None:
            signature.append((key, normalized))

    return tuple(signature)


def _deduplicate_numeric_rows(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
    """
    Remove rows that have exactly matching numeric values.
    Keeps the first occurrence and drops subsequent duplicates.
    """
    seen: Set[Tuple[Tuple[str, str], ...]] = set()
    deduped: List[Dict[str, str]] = []
    removed = 0

    for row in rows:
        signature = _build_numeric_signature(row)

        # If a row has no numeric values, keep it as-is.
        if not signature:
            deduped.append(row)
            continue

        if signature in seen:
            removed += 1
            continue

        seen.add(signature)
        deduped.append(row)

    return deduped, removed


def _get_all_fieldnames(records: List[Dict[str, str]]) -> List[str]:
    """Get all unique field names in order."""
    seen = set()
    result = []
    
    # Priority order for common columns
    priority = [
        'experiment_id', 'epoch', 'source_file', 'updated_at',
        'clip_style', 'clip_content', 'content_lpips',
        'all_clip_style', 'all_clip_content', 'all_content_lpips',
        'all_fid', 'all_art_fid', 'all_classifier_acc',
        'transfer_clip_style', 'transfer_clip_content', 'transfer_content_lpips',
        'transfer_fid', 'transfer_art_fid', 'transfer_classifier_acc',
        'photo_to_art_clip_style', 'photo_to_art_clip_content',
        'photo_to_art_fid', 'photo_to_art_art_fid',
        'photo_to_art_classifier_acc',
    ]
    
    # Add priority columns first
    for col in priority:
        if any(col in r for r in records):
            result.append(col)
            seen.add(col)
    
    # Add remaining columns
    for record in records:
        for key in record.keys():
            if key not in seen:
                result.append(key)
                seen.add(key)
    
    return result


def _has_clip_content(record: Dict[str, str]) -> bool:
    """Keep only rows that contain at least one clip_content metric."""
    for key in ('clip_content', 'all_clip_content', 'transfer_clip_content', 'photo_to_art_clip_content'):
        value = record.get(key, '')
        if value not in {'', None}:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Import summary/history json files to CSV with per-epoch metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single file
  %(prog)s --input summary_history.json --output metrics.csv
  
  # Directory (recursive by default)
  %(prog)s --input ./runs --output all_metrics.csv
  
  # Incremental update (append new records only)
  %(prog)s --input ./runs --output metrics.csv
        '''
    )
    ap.add_argument('--input', '-i', required=True, help='JSON file or directory')
    ap.add_argument('--output', '-o', required=False,default='summary.csv', help='Output CSV file')
    ap.add_argument('--recursive', '-r', action='store_true', help='Recursively search for JSON files (default for dirs)')
    ap.add_argument('--no-recursive', action='store_true', help='Disable recursive search when input is a directory')
    ap.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    ap.add_argument('--fresh', action='store_true', help='Ignore existing CSV and rebuild from current input only')
    ap.add_argument('--summary-only', action='store_true', help='Only parse files named summary.json (ignore summary_history*.json)')
    ap.add_argument(
        '--drop-missing-clip-content',
        action='store_true',
        help='Drop rows where both transfer_clip_content and photo_to_art_clip_content are empty',
    )
    
    args = ap.parse_args()
    
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    
    if not source.exists():
        raise SystemExit(f"❌ Source not found: {source}")
    
    # Find JSON files (directories default to recursive scan).
    recursive_scan = bool(args.recursive or source.is_dir())
    if args.no_recursive:
        recursive_scan = False
    json_files = _get_all_json_files(source, recursive=recursive_scan, summary_only=bool(args.summary_only))
    
    if not json_files:
        raise SystemExit(f"❌ No supported json files found in {source}")
    
    if args.verbose:
        print(f"[INFO] Found {len(json_files)} JSON file(s)")
        for jf in json_files:
            print(f"   - {jf}")
    
    # Extract all records
    all_records: List[Dict[str, Any]] = []
    
    for json_file in json_files:
        if args.verbose:
            print(f"[INFO] Processing {json_file}...")
        
        json_data = _read_json(json_file)
        records = _extract_rounds_data(json_data, json_file)
        if not records:
            records = _extract_single_summary_data(json_data, json_file)
        
        if args.verbose:
            print(f"   -> Extracted {len(records)} epoch record(s)")
        
        all_records.extend(records)
    
    if not all_records:
        raise SystemExit("❌ No records extracted from JSON files")
    
    trial_label_map = _build_optuna_trial_label_map(source)
    if args.verbose and trial_label_map:
        print(f"[INFO] Loaded {len(trial_label_map)} trial label(s) from optuna DB")

    # Flatten records for CSV
    flat_records = []
    for r in all_records:
        fr = _flatten_record(r)
        if trial_label_map:
            fr = _apply_trial_label_map(fr, trial_label_map)
        fr = _normalize_row_schema(fr)
        flat_records.append(fr)
    
    # Read existing CSV if it exists (unless fresh rebuild is requested)
    if args.fresh:
        existing_rows, existing_keys = [], set()
    else:
        existing_rows, existing_keys = _read_existing_csv(output)
    
    if args.verbose and existing_rows:
        print(f"[INFO] Found {len(existing_rows)} existing record(s)")
    
    # Merge with deduplication
    merged_rows, added_count = _merge_and_deduplicate(
        existing_rows,
        existing_keys,
        flat_records,
    )

    merged_rows, removed_numeric_duplicates = _deduplicate_numeric_rows(merged_rows)

    removed_missing_clip_content = 0
    if args.drop_missing_clip_content:
        removed_missing_clip_content = len(merged_rows)
        merged_rows = [row for row in merged_rows if _has_clip_content(row)]
        removed_missing_clip_content -= len(merged_rows)
    
    if args.verbose:
        print(f"[INFO] {added_count} new record(s) to add")
        if removed_numeric_duplicates > 0:
            print(f"[INFO] Removed {removed_numeric_duplicates} numeric-duplicate record(s)")
        if removed_missing_clip_content > 0:
            print(f"[INFO] Removed {removed_missing_clip_content} record(s) without clip_content")
    
    # Get all field names
    fieldnames = _get_all_fieldnames(merged_rows)
    
    # Write CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with output.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)
        
        print(f"[OK] Wrote {len(merged_rows)} total record(s) to {output}")
        if added_count > 0:
            print(f"   -> Added {added_count} new, kept {len(merged_rows) - added_count} existing")
        if removed_numeric_duplicates > 0:
            print(f"   -> Removed {removed_numeric_duplicates} row(s) with identical numeric values")
        if removed_missing_clip_content > 0:
            print(f"   -> Removed {removed_missing_clip_content} row(s) without clip_content")
        print(f"   -> {len(fieldnames)} column(s)")
        
    except Exception as e:
        raise SystemExit(f"❌ Error writing CSV: {e}")


if __name__ == '__main__':
    main()
