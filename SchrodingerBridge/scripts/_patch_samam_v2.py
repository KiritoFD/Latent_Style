"""Restore SaMam SS2D files from backup and apply clean patch."""
import shutil
from pathlib import Path

SAMAM_REPO = Path(r"I:\Github\Latent_Style\Related_Works\repos\SaMam")
TARGET_FILES = [
    SAMAM_REPO / "ARCHI" / "SS2D_Encoder.py",
    SAMAM_REPO / "ARCHI" / "SAVSSM" / "SS2D_Decoder.py",
]

for file in TARGET_FILES:
    bak = file.with_suffix(".py.bak")
    if bak.exists():
        shutil.copy2(bak, file)
        print(f"[RESTORED] {file} from backup")
    else:
        print(f"[NO BACKUP] {file}")

# Now apply clean patch: wrap any non-commented mamba_ssm import with try/except
for file in TARGET_FILES:
    if not file.exists():
        continue
    lines = file.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []
    patched_count = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # Skip comment lines
        if stripped.startswith("#"):
            new_lines.append(line)
            continue
        # Match non-commented mamba_ssm import
        if "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" in line:
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}try:\n")
            new_lines.append(f"{indent}    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn\n")
            new_lines.append(f"{indent}except ImportError:\n")
            new_lines.append(f"{indent}    from ARCHI.selective_scan_torch import selective_scan_fn\n")
            patched_count += 1
        else:
            new_lines.append(line)

    if patched_count > 0:
        file.write_text("".join(new_lines), encoding="utf-8")
        print(f"[PATCHED] {file.name}: {patched_count} import(s) wrapped")
    else:
        print(f"[NO CHANGE] {file.name}")

print("\nDone.")
