from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _walk_strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def test_active_modules_resolve_from_project_root() -> None:
    code = """
import json
from pathlib import Path
import config_schema
import model
import trainer
import utils.dataset

print(json.dumps({
    'config_schema': str(Path(config_schema.__file__).resolve()),
    'model': str(Path(model.__file__).resolve()),
    'trainer': str(Path(trainer.__file__).resolve()),
    'dataset': str(Path(utils.dataset.__file__).resolve()),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    paths = {name: Path(path) for name, path in json.loads(result.stdout).items()}
    assert paths["config_schema"].parent == PROJECT_ROOT
    assert paths["model"].parent == PROJECT_ROOT
    assert paths["trainer"].parent == PROJECT_ROOT
    assert paths["dataset"].parent == PROJECT_ROOT / "utils"


def test_root_training_entry_point_has_help() -> None:
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_canonical_configs_are_portable_and_complete() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from config_schema import (
        load_config,
        load_experiment_config,
        merge_config_dicts,
        resolve_full_eval_section,
        resolve_inference_section,
    )
    from model import build_model_from_config, count_parameters

    config_path = PROJECT_ROOT / "config.json"
    inference_path = PROJECT_ROOT / "inference.json"
    config = load_experiment_config(config_path)
    inference = load_config(inference_path)
    merged = merge_config_dicts(config.to_dict(), inference)
    inference_settings = resolve_inference_section(merged)
    eval_settings = resolve_full_eval_section(merged)

    assert count_parameters(build_model_from_config(config.model, bridge_cfg=config.bridge)) == 873_680
    assert inference_settings == {"num_steps": 8, "step_size": 1.0, "style_strength": 1.0}
    assert eval_settings["num_steps"] == 8
    assert eval_settings["batch_size"] == 16
    assert eval_settings["max_ref_compare"] == 16
    assert eval_settings["max_ref_cache"] == 16
    assert eval_settings["target_chunk_size"] == 1
    assert eval_settings["save_generated_images"] is True
    assert merged["model"]["endpoint_adain_scale"] == 2.0

    for path in (config_path, inference_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        bad_paths = [
            value for value in _walk_strings(payload)
            if WINDOWS_ABSOLUTE_PATH.match(value) or value.startswith(("/mnt/", "\\\\"))
        ]
        assert not bad_paths, f"absolute paths in {path.name}: {bad_paths}"


def test_submission_launchers_do_not_use_src_entry_points() -> None:
    launcher = (PROJECT_ROOT / "scripts" / "run_submission_repro.ps1").read_text(encoding="utf-8")
    evaluator = (PROJECT_ROOT / "scripts" / "batch_eval_all.py").read_text(encoding="utf-8")

    assert "src\\run.py" not in launcher
    assert "src\\default_config.json" not in launcher
    assert ' / "src" / "utils" ' not in evaluator


def test_epoch_selection_parser() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from batch_eval_all import parse_epoch_selection

    assert parse_epoch_selection("6") == {6}
    assert parse_epoch_selection("1,3-5,8") == {1, 3, 4, 5, 8}


def test_oriented_hf_route_changes_only_matching_hf_heads() -> None:
    import torch

    sys.path.insert(0, str(PROJECT_ROOT))
    from config_schema import load_experiment_config
    from model import build_model_from_config

    config = load_experiment_config(PROJECT_ROOT / "experiments" / "architecture" / "hf_oriented_nohh.json")
    model = build_model_from_config(config.model, bridge_cfg=config.bridge).eval()
    source = torch.linspace(-1.0, 1.0, 4 * 16 * 16).reshape(1, 4, 16, 16)
    style_a = torch.flip(source, dims=(-1,))
    style_b = torch.flip(source, dims=(-2,))

    with torch.inference_mode():
        output_a = model(source, t=0.5, style_id=2, style_latent=style_a)
        output_b = model(source, t=0.5, style_id=2, style_latent=style_b)

    assert output_a.keys() == {"ll", "lh", "hl"}
    assert torch.equal(output_a["ll"], output_b["ll"])
    assert not torch.equal(output_a["lh"], output_b["lh"])
    assert not torch.equal(output_a["hl"], output_b["hl"])
