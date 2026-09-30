"""Patch inference.py to use bf16 VAE by default."""
import sys

target = r"I:\Github\Latent_Style\WEAVE\utils\inference.py"
with open(target, "r", encoding="utf-8") as f:
    content = f.read()

old = "compile_dtype: torch.dtype = torch.float16"
new = "compile_dtype: torch.dtype = torch.bfloat16"

if old in content:
    content = content.replace(old, new)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Patched: {old} -> {new}")
else:
    print(f"Pattern not found in {target}")
    # Check what's there
    for i, line in enumerate(content.splitlines()):
        if "compile_dtype" in line:
            print(f"  Line {i+1}: {line.strip()}")