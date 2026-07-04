#!/bin/bash
echo "=== Copying updated source files to WSL project ==="
SRC_WIN="/mnt/g/GitHub/Latent_Style/SchrodingerBridge/src"
DST_WSL="/home/xy/Latent_Style/SchrodingerBridge/src"

FILES="style_families.py model620.py config_schema.py losses620.py blocks620.py trainer.py"

for FILE in $FILES; do
    if [ -f "$SRC_WIN/$FILE" ]; then
        echo "Copying $FILE..."
        cp "$SRC_WIN/$FILE" "$DST_WSL/$FILE"
        echo "  ✓ $FILE copied ($(du -h $DST_WSL/$FILE | cut -f1))"
    else
        echo "  ✗ $FILE NOT FOUND at source"
    fi
done

echo ""
echo "=== Verifying style_families.py has validate_phase616_clean_contract ==="
if grep -q "validate_phase616_clean_contract" /home/xy/Latent_Style/SchrodingerBridge/src/style_families.py; then
    echo "✓ validate_phase616_clean_contract FOUND in style_families.py"
else
    echo "✗ validate_phase616_clean_contract NOT FOUND"
fi