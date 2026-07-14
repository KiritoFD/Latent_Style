from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
