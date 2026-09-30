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
DEFAULT_OUTPUT_DIR = SB_ROOT / "configs" / "aaai2027" / "round1_full_sweep" / "warmstart"


def _load_manifest_row(manifest_csv: Path, *, family_id: str) -> dict[str, str]:
    rows = read_csv_rows(manifest_csv)
    for row in rows:
        if str(row.get("family_id", "")).strip() == str(family_id).strip():
            return row
    raise KeyError(f"family_id not found in manifest: {family_id}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a round-1 tokenizer warm-start config using the distillation/tokenizer-branch path.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--distill-mode", choices=["tokenizer_only", "style_branch"], default="style_branch")
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-weight", type=float, default=0.0)
    parser.add_argument("--reinit-trainable", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    row = _load_manifest_row(manifest_csv, family_id=str(args.family_id))
    axis = str(row.get("axis", "")).strip().lower()
    if axis != "tokenizer":
        raise ValueError(f"Warm-start prep is intended for tokenizer families only, got axis={axis} for {args.family_id}")

    config_path = Path(str(row.get("config_path", "")).strip())
    if not config_path.is_absolute():
        config_path = (WORKSPACE / config_path).resolve()
    payload = deepcopy(_load_json(config_path))

    teacher_ckpt = str(args.teacher_checkpoint).strip()
    if not teacher_ckpt:
        teacher_ckpt = str(((payload.get("training") or {}).get("resume_checkpoint", ""))).strip()
    if not teacher_ckpt:
        raise ValueError("No teacher checkpoint available. Pass --teacher-checkpoint or ensure training.resume_checkpoint exists in the base family config.")

    train_cfg = payload.setdefault("training", {})
    train_cfg["distill"] = {
        "enabled": True,
        "teacher_checkpoint": teacher_ckpt,
        "mode": str(args.distill_mode),
        "velocity_weight": float(args.velocity_weight),
        "endpoint_weight": float(args.endpoint_weight),
        "reinit_trainable": bool(args.reinit_trainable),
    }
    train_cfg["resume_checkpoint"] = ""
    train_cfg["resume_training_state"] = False
    train_cfg["resume_optimizer"] = False
    train_cfg["num_epochs"] = int(args.num_epochs)
    train_cfg["save_interval"] = int(args.save_interval)
    train_cfg["freeze_mode"] = str(args.distill_mode)

    run_name = f"aaai2027_round1_{str(args.family_id).strip()}_warmstart_seed42_b8a2"
    payload.setdefault("checkpoint", {})
    payload["checkpoint"]["save_dir"] = f"./exp/inmortal-exp/{run_name}"
    payload.setdefault("ablation", {})
    payload["ablation"]["name"] = run_name
    payload["ablation"]["axis"] = "aaai2027_round1_tokenizer_warmstart"
    payload["ablation"]["stage"] = "round1_tokenizer_warmstart"
    payload["ablation"]["notes"] = (
        f"Tokenizer warm-start for {args.family_id} using distill mode={args.distill_mode}, "
        f"teacher={teacher_ckpt}"
    )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (WORKSPACE / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_name}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    doc_dir = (WORKSPACE / ROUND1_DOC_DIR).resolve() / str(args.family_id).strip()
    doc_dir.mkdir(parents=True, exist_ok=True)
    warmstart_note = doc_dir / "warmstart.md"
    warmstart_note.write_text(
        "\n".join(
            [
                f"# {args.family_id} Warmstart",
                "",
                f"- Base config: `{config_path}`",
                f"- Warmstart config: `{out_path}`",
                f"- Distill mode: `{args.distill_mode}`",
                f"- Teacher checkpoint: `{teacher_ckpt}`",
                f"- Num epochs: `{int(args.num_epochs)}`",
                f"- Save interval: `{int(args.save_interval)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(out_path)
    print(warmstart_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
