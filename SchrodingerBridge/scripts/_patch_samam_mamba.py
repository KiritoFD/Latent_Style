"""Patch SS2D_Encoder.py to make mamba_ssm import optional with try/except fallback."""
import shutil
from pathlib import Path

file = Path(r"I:\Github\Latent_Style\Related_Works\repos\SaMam\ARCHI\SS2D_Encoder.py")

# Backup
shutil.copy2(file, str(file) + ".bak")

lines = file.read_text(encoding="utf-8").splitlines(keepends=True)
patched = False

new_lines = []
for i, line in enumerate(lines):
    stripped = line.rstrip()
    # Match the mamba_ssm import line inside the if mamba_from_trion: block
    if (
        "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" in stripped
        and not patched
        and i > 0
        and "if mamba_from_trion" in lines[i - 1]
    ):
        # Get indentation
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(f"{indent}try:\n")
        new_lines.append(f"{indent}    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn\n")
        new_lines.append(f"{indent}except ImportError:\n")
        new_lines.append(f"{indent}    from ARCHI.selective_scan_torch import selective_scan_fn\n")
        patched = True
        print(f"Patched line {i}: {stripped}")
    else:
        new_lines.append(line)

if patched:
    file.write_text("".join(new_lines), encoding="utf-8")
    print(f"File saved: {file}")
else:
    print("No matching pattern found.")

# Verify
print("\n=== First 45 lines after patch ===")
for i, line in enumerate(file.read_text(encoding="utf-8").splitlines()[:45], 1):
    print(f"{i:3d}: {line}")
