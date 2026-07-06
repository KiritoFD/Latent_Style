"""Patch all SaMam SS2D files to make mamba_ssm import optional."""
import shutil
from pathlib import Path

SAMAM_REPO = Path(r"I:\Github\Latent_Style\Related_Works\repos\SaMam")
TARGET_FILES = [
    SAMAM_REPO / "ARCHI" / "SS2D_Encoder.py",
    SAMAM_REPO / "ARCHI" / "SAVSSM" / "SS2D_Decoder.py",
]

for file in TARGET_FILES:
    if not file.exists():
        print(f"[SKIP] {file} not found")
        continue

    # Backup
    if not file.with_suffix(".py.bak").exists():
        shutil.copy2(file, str(file) + ".bak")

    lines = file.read_text(encoding="utf-8").splitlines(keepends=True)
    patched = False
    new_lines = []

    for i, line in enumerate(lines):
        stripped = line.rstrip()
        # Match the mamba_ssm import line
        if (
            "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" in stripped
            and not patched
        ):
            indent = line[:len(line) - len(line.lstrip())]
            # Check context: is it inside if mamba_from_trion?
            prev_line = lines[i - 1].strip() if i > 0 else ""
            if "if mamba_from_trion" in prev_line:
                new_lines.append(f"{indent}try:\n")
                new_lines.append(f"{indent}    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn\n")
                new_lines.append(f"{indent}except ImportError:\n")
                new_lines.append(f"{indent}    from ARCHI.selective_scan_torch import selective_scan_fn\n")
                patched = True
                print(f"[PATCHED] {file.name} line {i+1}")
            else:
                # Standalone import — wrap with try/except
                new_lines.append(f"{indent}try:\n")
                new_lines.append(f"{indent}    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn\n")
                new_lines.append(f"{indent}except ImportError:\n")
                new_lines.append(f"{indent}    from ARCHI.selective_scan_torch import selective_scan_fn\n")
                patched = True
                print(f"[PATCHED] {file.name} line {i+1} (standalone)")
        else:
            new_lines.append(line)

    if patched:
        file.write_text("".join(new_lines), encoding="utf-8")
        print(f"[SAVED] {file}")
    else:
        print(f"[NO CHANGE] {file}")

print("\nDone.")
