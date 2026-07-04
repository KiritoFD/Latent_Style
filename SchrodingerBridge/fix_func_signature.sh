#!/bin/bash
echo "=== Fixing validate_pure_latent_contract signature ==="

# Create a Python script to fix the function signature
python3 << 'PYEOF'
import re

file_path = '/home/xy/Latent_Style/SchrodingerBridge/src/style_families.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find and replace the function signature
old_sig = '''def validate_pure_latent_contract(
    *,
    tokenizer_family: str,
    style_tokenizer: str = "",
    semantic_supervision_family: str = "",
    dino_masked_swd_weight: float = 0.0,
    tokenizer_content_adaptive: bool = False,
) -> None:'''

new_sig = '''def validate_pure_latent_contract(
    *,
    tokenizer_family: str,
    style_tokenizer: str = "",
    semantic_supervision_family: str = "",
    dino_masked_swd_weight: float = 0.0,
    style_spatial_mode: str = "",
    tokenizer_content_adaptive: bool = False,
) -> None:'''

if old_sig in content:
    content = content.replace(old_sig, new_sig)
    with open(file_path, 'w') as f:
        f.write(content)
    print("✓ Added style_spatial_mode parameter to validate_pure_latent_contract")
else:
    print("✗ Could not find the exact function signature to replace")
    print("Attempting regex replacement...")
    
    # Try regex approach
    pattern = r'(def validate_pure_latent_contract\(\s*\*,\s*tokenizer_family: str,\s*style_tokenizer: str = "",\s*semantic_supervision_family: str = "",\s*dino_masked_swd_weight: float = 0\.0,)(\s*tokenizer_content_adaptive: bool = False,\s*\) -> None:)'
    replacement = r'\1\n    style_spatial_mode: str = "",\2'
    
    new_content, count = re.sub(pattern, replacement, content)
    if count > 0:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"✓ Fixed using regex (replaced {count} occurrence)")
    else:
        print("✗ Regex replacement also failed")

PYEOF

echo ""
echo "=== Verifying fix ==="
grep -A 8 "^def validate_pure_latent_contract" /home/xy/Latent_Style/SchrodingerBridge/src/style_families.py | head -10