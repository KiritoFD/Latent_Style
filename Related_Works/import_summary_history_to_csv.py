#!/usr/bin/env python3
"""
Import summary_history.json data to CSV with per-epoch metrics.
Supports incremental updates with deduplication.

Usage:
    python import_summary_history_to_csv.py --input <file_or_dir> --output output.csv
    python import_summary_history_to_csv.py --input . --output metrics.csv --recursive
"""

import argparse
import csv
import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


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
    parts = normalized.split('/')
    
    # Find the directory before "full_eval", "eval", "epoch_*", or "summary.json"
    for i, part in enumerate(parts):
        if part in {'full_eval', 'eval'} or part.startswith('epoch_'):
            if i > 0:
                exp_name = parts[i - 1].strip()
                if exp_name and exp_name not in {'.', '..'}:
                    return exp_name
    
    # Fallback: return first non-trivial directory name
    for part in parts:
        if part and part not in {'.', '..', 'summary.json'}:
            return part
    
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


def _extract_experiment_id_from_file_path(path: Path) -> str:
    parts = list(path.parts)
    for i, part in enumerate(parts):
        if part == 'full_eval' and i > 0:
            candidate = parts[i - 1].strip()
            if candidate:
                return candidate
    return path.parent.name or 'unknown'


def _extract_single_summary_data(json_data: Dict[str, Any], source_path: Path) -> List[Dict[str, Any]]:
    """
    Extract one row from full_eval/epoch_xxxx/summary.json.
    """
    analysis = json_data.get('analysis')
    if not isinstance(analysis, dict):
        return []

    transfer = analysis.get('style_transfer_ability')
    photo_to_art = analysis.get('photo_to_art_performance')
    if not isinstance(transfer, dict) or not isinstance(photo_to_art, dict):
        return []

    epoch = _extract_epoch_from_path(source_path)
    record = {
        'source_file': str(source_path),
        'experiment_id': _extract_experiment_id_from_file_path(source_path),
        'updated_at': json_data.get('timestamp', ''),
        'summary_path': str(source_path),
        'epoch': int(epoch) if epoch is not None else '',
        'transfer_clip_style': transfer.get('clip_style'),
        'transfer_content_lpips': transfer.get('content_lpips'),
        'transfer_fid': transfer.get('fid'),
        'transfer_art_fid': transfer.get('art_fid'),
        'transfer_classifier_acc': transfer.get('classifier_acc'),
        'photo_to_art_clip_style': photo_to_art.get('clip_style'),
        'photo_to_art_fid': photo_to_art.get('fid'),
        'photo_to_art_art_fid': photo_to_art.get('art_fid'),
        'photo_to_art_classifier_acc': photo_to_art.get('classifier_acc'),
    }
    return [record]


def _get_all_json_files(source: Path, recursive: bool = False) -> List[Path]:
    """Find all supported json files (summary_history*.json and summary.json)."""
    if source.is_file():
        if source.suffix == '.json' and (source.name.startswith('summary_history') or source.name == 'summary.json'):
            return [source]
        return []
    
    if not source.is_dir():
        return []
    
    pattern = '**/*.json' if recursive else '*.json'
    return list(source.glob(pattern))


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


def _get_dedup_key(record: Dict[str, str]) -> str:
    """Generate key for deduplication."""
    exp_id = record.get('experiment_id', 'unknown')
    epoch = record.get('epoch', '0')
    source = record.get('source_file', '')
    return f"{exp_id}|{epoch}|{source}"


def _read_existing_csv(csv_path: Path) -> Tuple[List[Dict[str, str]], Set[str]]:
    """Read existing CSV and extract dedup keys."""
    if not csv_path.exists():
        return [], set()
    
    rows = []
    keys = set()
    
    try:
        with csv_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return [], set()
            
            for row in reader:
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


def _get_all_fieldnames(records: List[Dict[str, str]]) -> List[str]:
    """Get all unique field names in order."""
    seen = set()
    result = []
    
    # Priority order for common columns
    priority = [
        'experiment_id', 'epoch', 'source_file', 'updated_at',
        'transfer_clip_style', 'transfer_content_lpips',
        'transfer_fid', 'transfer_art_fid', 'transfer_classifier_acc',
        'photo_to_art_clip_style', 'photo_to_art_fid', 'photo_to_art_art_fid',
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Import summary_history.json files to CSV with per-epoch metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single file
  %(prog)s --input summary_history.json --output metrics.csv
  
  # Directory (non-recursive)
  %(prog)s --input ./runs --output all_metrics.csv
  
  # Directory (recursive)
  %(prog)s --input ./runs --output all_metrics.csv --recursive
  
  # Incremental update (append new records only)
  %(prog)s --input ./runs --output metrics.csv --recursive
        '''
    )
    ap.add_argument('--input', '-i', required=True, help='JSON file or directory')
    ap.add_argument('--output', '-o', required=False,default='summary.csv', help='Output CSV file')
    ap.add_argument('--recursive', '-r', action='store_true', help='Recursively search for JSON files')
    ap.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = ap.parse_args()
    
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    
    if not source.exists():
        raise SystemExit(f"❌ Source not found: {source}")
    
    # Find JSON files
    json_files = _get_all_json_files(source, recursive=args.recursive)
    
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
    
    # Flatten records for CSV
    flat_records = [_flatten_record(r) for r in all_records]
    
    # Read existing CSV if it exists
    existing_rows, existing_keys = _read_existing_csv(output)
    
    if args.verbose and existing_rows:
        print(f"[INFO] Found {len(existing_rows)} existing record(s)")
    
    # Merge with deduplication
    merged_rows, added_count = _merge_and_deduplicate(
        existing_rows,
        existing_keys,
        flat_records,
    )
    
    if args.verbose:
        print(f"[INFO] {added_count} new record(s) to add")
    
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
        print(f"   -> {len(fieldnames)} column(s)")
        
    except Exception as e:
        raise SystemExit(f"❌ Error writing CSV: {e}")


if __name__ == '__main__':
    main()
