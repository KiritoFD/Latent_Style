from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round2_pure_sde" / "round2_family_manifest.csv"
DEFAULT_OUTPUT_ROOT = SB_ROOT / "configs" / "aaai2027" / "round2_pure_sde" / "followon"
DEFAULT_DOC_ROOT = SB_ROOT / "docs" / "experiments" / "round2_pure_sde" / "followon"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        clean: dict[str, str] = {}
        for key, value in raw.items():
            if key is None:
                continue
            normalized = str(key).replace("\ufeff", "").replace('"', "").strip()
            clean[normalized] = value
        rows.append(clean)
    return rows


def _find_row(rows: list[dict[str, str]], family_id: str) -> dict[str, str]:
    target = str(family_id).strip()
    for row in rows:
        if str(row.get("family_id", "")).strip() == target:
            return row
    raise KeyError(f"family_id not found in manifest: {family_id}")


def _winner_checkpoint_from_row(row: dict[str, str], *, mode: str) -> str:
    selected = str(mode).strip().lower()
    field_map = {
        "latest": "latest_checkpoint",
        "best_transfer": "best_transfer_epoch",
        "best_all_pairs": "best_all_pairs_epoch",
    }
    if selected == "latest":
        checkpoint = str(row.get("latest_checkpoint", "")).strip()
        if checkpoint:
            return checkpoint
    elif selected == "best_transfer":
        epoch = str(row.get("best_transfer_epoch", "")).strip()
        run_dir = str(row.get("run_dir", "")).strip()
        if epoch and run_dir:
            return f"{run_dir.rstrip('/')}/{epoch}.pt"
    elif selected == "best_all_pairs":
        epoch = str(row.get("best_all_pairs_epoch", "")).strip()
        run_dir = str(row.get("run_dir", "")).strip()
        if epoch and run_dir:
            return f"{run_dir.rstrip('/')}/{epoch}.pt"
    raise ValueError(
        f"Could not infer winner checkpoint for family_id={row.get('family_id', '')} using mode={mode}. "
        f"Manifest fields available: {', '.join(f'{k}={row.get(v, '')}' for k, v in field_map.items())}"
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _remote_checkpoint_path(raw: str, *, remote_wsl_cwd: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("winner checkpoint path is empty")
    if text.startswith("/"):
        return text
    if text.startswith("./"):
        return f"{remote_wsl_cwd.rstrip('/')}/{text[2:]}"
    return f"{remote_wsl_cwd.rstrip('/')}/{text}"


def _checkpoint_epoch_token(checkpoint_path: str) -> str:
    stem = Path(checkpoint_path).stem
    return stem if stem else "checkpoint"


def _sanitize_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text).strip())


def _strip_existing_followon_suffix(name: str) -> str:
    text = str(name or "").strip()
    marker = "_from_"
    idx = text.find(marker)
    if idx >= 0:
        return text[:idx]
    return text


def _target_rows(
    rows: list[dict[str, str]],
    *,
    family_ids: list[str],
    wave: str,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    wanted = {str(x).strip() for x in family_ids if str(x).strip()}
    for row in rows:
        fid = str(row.get("family_id", "")).strip()
        if wanted and fid not in wanted:
            continue
        if (not wanted) and str(row.get("wave", "")).strip() != str(wave).strip():
            continue
        out.append(row)
    if not out:
        raise ValueError("No target families matched the requested selection.")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare round-2 follow-on configs that warm-start solver/loss waves from a tokenizer winner checkpoint."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--winner-family-id", required=True)
    parser.add_argument("--winner-checkpoint", default="")
    parser.add_argument(
        "--winner-checkpoint-mode",
        choices=["latest", "best_transfer", "best_all_pairs"],
        default="latest",
    )
    parser.add_argument("--target-wave", default="wave2_sde_noise")
    parser.add_argument("--target-family-id", action="append", default=[])
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--doc-root", type=Path, default=DEFAULT_DOC_ROOT)
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    rows = _read_rows(manifest_csv)
    winner = _find_row(rows, str(args.winner_family_id))

    checkpoint_raw = str(args.winner_checkpoint).strip()
    if not checkpoint_raw:
        checkpoint_raw = _winner_checkpoint_from_row(winner, mode=str(args.winner_checkpoint_mode))
    checkpoint_path = _remote_checkpoint_path(checkpoint_raw, remote_wsl_cwd=str(args.remote_wsl_cwd))
    epoch_token = _checkpoint_epoch_token(checkpoint_path)
    winner_token = _sanitize_token(str(args.winner_family_id))
    suffix = f"from_{winner_token}_{epoch_token}"

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (WORKSPACE / output_root).resolve()
    output_root = output_root / winner_token
    output_root.mkdir(parents=True, exist_ok=True)

    doc_root = Path(args.doc_root).expanduser()
    if not doc_root.is_absolute():
        doc_root = (WORKSPACE / doc_root).resolve()
    doc_root = doc_root / winner_token
    doc_root.mkdir(parents=True, exist_ok=True)

    targets = _target_rows(
        rows,
        family_ids=list(args.target_family_id),
        wave=str(args.target_wave),
    )

    produced: list[dict[str, str]] = []
    for row in targets:
        config_path = Path(str(row.get("config_path", "")).strip())
        if not config_path.is_absolute():
            config_path = (WORKSPACE / config_path).resolve()
        payload = deepcopy(_load_json(config_path))
        payload.setdefault("training", {})
        payload["training"]["resume_checkpoint"] = checkpoint_path
        payload["training"]["resume_prefer_local_checkpoint"] = False
        payload["training"]["resume_training_state"] = False
        payload["training"]["resume_optimizer"] = False
        payload["training"]["resume_model_strict"] = True
        payload["training"]["resume_ignore_prefixes"] = []
        payload["training"]["resume_include_prefixes"] = []

        base_name = str((payload.get("ablation") or {}).get("name", Path(config_path).stem)).strip() or Path(config_path).stem
        base_name = _strip_existing_followon_suffix(base_name)
        followon_name = f"{base_name}_{suffix}"
        payload.setdefault("training", {})
        payload["training"]["remote_log_name"] = followon_name
        payload.setdefault("checkpoint", {})
        payload["checkpoint"]["save_dir"] = f"./exp/inmortal-exp/{followon_name}"
        payload.setdefault("ablation", {})
        payload["ablation"]["name"] = followon_name
        notes = str(payload["ablation"].get("notes", "")).strip()
        addendum = f" Warm-start from {args.winner_family_id} {epoch_token}."
        payload["ablation"]["notes"] = (notes + addendum).strip()

        out_path = output_root / f"{followon_name}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        produced.append(
            {
                "family_id": str(row.get("family_id", "")).strip(),
                "config_path": str(out_path),
                "resume_checkpoint": checkpoint_path,
                "run_name": followon_name,
            }
        )
        print(out_path)

    note_path = doc_root / f"{str(args.target_wave).strip()}_{suffix}.md"
    lines = [
        f"# Round2 Follow-on: {args.target_wave}",
        "",
        f"- Winner family: `{args.winner_family_id}`",
        f"- Winner checkpoint: `{checkpoint_path}`",
        f"- Source manifest: `{manifest_csv}`",
        "",
        "## Generated Configs",
    ]
    for item in produced:
        lines.extend(
            [
                f"- {item['family_id']}:",
                f"  - Config: `{item['config_path']}`",
                f"  - Run name: `{item['run_name']}`",
            ]
        )
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(note_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
