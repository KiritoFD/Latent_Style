from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from csv_utils import read_csv_rows
from round1_registry import ROUND1_DOC_DIR


DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"
DEFAULT_OUTPUT_DIR = SB_ROOT / "configs" / "aaai2027" / "round1_full_sweep" / "pretrain"


def _load_manifest_row(manifest_csv: Path, *, family_id: str) -> dict[str, str]:
    rows = read_csv_rows(manifest_csv)
    for row in rows:
        if str(row.get("family_id", "")).strip() == str(family_id).strip():
            return row
    raise KeyError(f"family_id not found in manifest: {family_id}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a round-1 tokenizer reconstruction-pretrain config using identity-only batches.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--freeze-mode", choices=["tokenizer_only", "style_branch"], default="style_branch")
    parser.add_argument("--terminal-swd-weight", type=float, default=12.0)
    parser.add_argument("--target-teacher-weight", type=float, default=0.0)
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    row = _load_manifest_row(manifest_csv, family_id=str(args.family_id))
    axis = str(row.get("axis", "")).strip().lower()
    if axis != "tokenizer":
        raise ValueError(f"Reconstruction pretrain prep is intended for tokenizer families only, got axis={axis} for {args.family_id}")

    config_path = Path(str(row.get("config_path", "")).strip())
    if not config_path.is_absolute():
        config_path = (WORKSPACE / config_path).resolve()
    payload = deepcopy(_load_json(config_path))

    train_cfg = payload.setdefault("training", {})
    train_cfg["freeze_mode"] = str(args.freeze_mode)
    train_cfg["resume_checkpoint"] = ""
    train_cfg["resume_training_state"] = False
    train_cfg["resume_optimizer"] = False
    train_cfg["num_epochs"] = int(args.num_epochs)
    train_cfg["save_interval"] = int(args.save_interval)

    bridge_cfg = payload.setdefault("bridge", {})
    bridge_cfg["identity_endpoint"] = True
    bridge_cfg["terminal_swd_on_identity"] = True
    bridge_cfg["terminal_swd_weight"] = float(args.terminal_swd_weight)
    bridge_cfg["target_teacher_mode"] = "off"
    bridge_cfg["target_teacher_weight"] = float(args.target_teacher_weight)
    bridge_cfg["cycle_consistency_weight"] = 0.0

    data_cfg = payload.setdefault("data", {})
    data_cfg["identity_ratio"] = 1.0
    data_cfg["pairing_cache_cross_only"] = False
    data_cfg["pairing_cache_dual_target_mix"] = 0.0
    data_cfg["pairing_cache_aux_target_topk"] = 0

    run_name = f"aaai2027_round1_{str(args.family_id).strip()}_reconpretrain_seed42_b8a2"
    payload.setdefault("checkpoint", {})
    payload["checkpoint"]["save_dir"] = f"./exp/inmortal-exp/{run_name}"
    payload.setdefault("ablation", {})
    payload["ablation"]["name"] = run_name
    payload["ablation"]["axis"] = "aaai2027_round1_tokenizer_reconstruction_pretrain"
    payload["ablation"]["stage"] = "round1_tokenizer_reconstruction_pretrain"
    payload["ablation"]["notes"] = (
        f"Identity-only tokenizer reconstruction pretrain for {args.family_id} "
        f"with freeze_mode={args.freeze_mode}"
    )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (WORKSPACE / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_name}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    doc_dir = (WORKSPACE / ROUND1_DOC_DIR).resolve() / str(args.family_id).strip()
    doc_dir.mkdir(parents=True, exist_ok=True)
    note_path = doc_dir / "reconstruction_pretrain.md"
    note_path.write_text(
        "\n".join(
            [
                f"# {args.family_id} Reconstruction Pretrain",
                "",
                f"- Base config: `{config_path}`",
                f"- Pretrain config: `{out_path}`",
                f"- Freeze mode: `{args.freeze_mode}`",
                f"- Identity ratio: `1.0`",
                f"- identity_endpoint: `True`",
                f"- terminal_swd_on_identity: `True`",
                f"- terminal_swd_weight: `{float(args.terminal_swd_weight)}`",
                f"- Num epochs: `{int(args.num_epochs)}`",
                f"- Save interval: `{int(args.save_interval)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(out_path)
    print(note_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
