#!/bin/bash
echo "=== Updating utils/training.py ==="
cp /mnt/c/Users/xy/training.py /home/xy/Latent_Style/SchrodingerBridge/src/utils/training.py
echo "✓ training.py updated"

echo ""
echo "=== Re-testing src/run.py import ==="
cd /home/xy/Latent_Style/SchrodingerBridge
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/xy/Latent_Style/SchrodingerBridge/src')
try:
    import run
    print("✓ src/run.py imported successfully")
    print(f"  - Has main: {hasattr(run, 'main')}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
PYEOF

echo ""
echo "=== Complete ==="