from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from round1_registry import ROUND1_DOC_DIR, ROUND1_FAMILY_SPECS


_FAMILY_IDS = tuple(sorted((spec.family_id for spec in ROUND1_FAMILY_SPECS), key=len, reverse=True))


def infer_round1_family_id(*, run_name: str = "", config_stem: str = "") -> str | None:
    text_candidates = [str(run_name).strip(), str(config_stem).strip()]
    for text in text_candidates:
        if not text:
            continue
        for family_id in _FAMILY_IDS:
            token = f"round1_{family_id}_"
            if token in text or text == family_id:
                return family_id
    return None


def round1_fast_local_root(*, family_id: str | None, run_name: str = "") -> Path:
    if family_id:
        return SB_ROOT / "aaai2027" / f"round1_{family_id}_fast_local"
    return SB_ROOT / "aaai2027" / f"{str(run_name).strip()}_fast_local"


def round1_localreview_root(*, family_id: str | None, run_name: str = "") -> Path:
    if family_id:
        return SB_ROOT / "aaai2027" / f"round1_{family_id}_localreview"
    return SB_ROOT / "aaai2027" / f"{str(run_name).strip()}_localreview"


def round1_family_doc_dir(*, family_id: str | None, run_name: str = "") -> Path:
    family_token = str(family_id).strip() or str(run_name).strip()
    return (SB_ROOT.parent / ROUND1_DOC_DIR).resolve() / family_token


def round1_switch_smoke_artifact(*, family_id: str | None, run_name: str = "") -> Path:
    family_token = str(family_id).strip() or str(run_name).strip()
    return SB_ROOT / "aaai2027" / f"round1_{family_token}_switch_smoke_latest.json"
