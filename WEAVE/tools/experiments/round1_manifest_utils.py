from __future__ import annotations

import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"


def resolve_manifest_csv(path: Path | str) -> Path:
    manifest_csv = Path(path).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    return manifest_csv


def status_of(row: dict[str, str]) -> str:
    return str(row.get("decision_status", "")).strip().lower()


def smoke_status_of(row: dict[str, str]) -> str:
    return str(row.get("switch_smoke_status", "")).strip().lower()


def config_families(row: dict[str, str]) -> tuple[str, str]:
    tokenizer_family = str(row.get("tokenizer_family", "")).strip().lower()
    semantic_supervision_family = str(row.get("semantic_supervision_family", "")).strip().lower()
    if tokenizer_family or semantic_supervision_family:
        return tokenizer_family, semantic_supervision_family

    config_path = Path(str(row.get("config_path", "")).strip())
    if not config_path:
        return tokenizer_family, semantic_supervision_family
    if not config_path.is_absolute():
        config_path = (WORKSPACE / config_path).resolve()
    if not config_path.is_file():
        return tokenizer_family, semantic_supervision_family
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return tokenizer_family, semantic_supervision_family

    model_cfg = payload.get("model") or {}
    bridge_cfg = payload.get("bridge") or {}
    tokenizer_family = tokenizer_family or str(model_cfg.get("tokenizer_family", "")).strip().lower()
    semantic_supervision_family = semantic_supervision_family or str(bridge_cfg.get("semantic_supervision_family", "")).strip().lower()
    return tokenizer_family, semantic_supervision_family


def is_dino_tail(row: dict[str, str]) -> bool:
    tokenizer_family, semantic_supervision_family = config_families(row)
    family_id = str(row.get("family_id", "")).strip().lower()
    return (
        "dino" in tokenizer_family
        or "dino" in semantic_supervision_family
        or family_id in {"tok_a_dino_dict", "tok_b_cross_image"}
    )


def rows_by_status(rows: list[dict[str, str]], *, status: str) -> list[dict[str, str]]:
    wanted = str(status).strip().lower()
    return [row for row in rows if status_of(row) == wanted]


def candidate_ids(rows: list[dict[str, str]]) -> list[str]:
    return [str(row.get("family_id", "")).strip() for row in rows if str(row.get("family_id", "")).strip()]


def relaunchable_non_dino(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if is_dino_tail(row):
            continue
        if status_of(row) != "recalibration_needed":
            continue
        if smoke_status_of(row) != "ok":
            continue
        out.append(row)
    return out
