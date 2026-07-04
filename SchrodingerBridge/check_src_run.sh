#!/bin/bash
echo "=== Checking src/run.py imports ==="
head -30 /home/xy/Latent_Style/SchrodingerBridge/src/run.py
echo ""
echo "=== Testing direct import of src.run ==="
cd /home/xy/Latent_Style/SchrodingerBridge
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/xy/Latent_Style/SchrodingerBridge/src')
try:
    import run
    print("✓ src/run.py imported successfully")
    print(f"  - Has main: {hasattr(run, 'main')}")
except Exception as e:
    print(f"✗ Failed to import src/run.py: {e}")
    import traceback
    traceback.print_exc()
PYEOF