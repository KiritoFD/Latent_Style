from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "experiments" / "launch_remote_wsl_command.py"
SPEC = importlib.util.spec_from_file_location("launch_remote_wsl_command", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
REMOTE_LAUNCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REMOTE_LAUNCH)


def test_hash_manifest_for_base_reads_expected_files(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    file_a = tmp_path / "a.py"
    file_b = tmp_path / "pkg" / "b.py"
    file_a.write_text("print('a')\n", encoding="utf-8")
    file_b.write_text("print('b')\n", encoding="utf-8")

    manifest = REMOTE_LAUNCH._hash_manifest_for_base(
        tmp_path,
        [Path("a.py"), Path("pkg/b.py")],
    )

    assert manifest == {
        "a.py": hashlib.sha256(file_a.read_bytes()).hexdigest(),
        "pkg/b.py": hashlib.sha256(file_b.read_bytes()).hexdigest(),
    }


def test_compare_hash_manifests_reports_mismatch_and_missing() -> None:
    issues = REMOTE_LAUNCH._compare_hash_manifests(
        {
            "a.py": "aaa",
            "b.py": "bbb",
        },
        {
            "a.py": "aaa",
            "b.py": "__missing__",
            "c.py": "ccc",
        },
    )

    assert issues == [
        {
            "path": "b.py",
            "local": "bbb",
            "remote": "__missing__",
        },
        {
            "path": "c.py",
            "local": "",
            "remote": "ccc",
        },
    ]
