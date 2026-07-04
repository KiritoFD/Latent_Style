#!/bin/bash
echo "=== Testing with src/ in path ==="
cd /home/xy/Latent_Style/SchrodingerBridge

# Test 1: Add src to path directly
echo "Test 1: Adding src/ to sys.path..."
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/xy/Latent_Style/SchrodingerBridge/src')
try:
    from config_schema import *
    print('✓ CONFIG_SCHEMA OK (with src in path)')
except Exception as e:
    print(f'✗ CONFIG_SCHEMA FAILED: {e}')

try:
    from model620 import *
    print('✓ MODEL OK (with src in path)')
except Exception as e:
    print(f'✗ MODEL FAILED: {e}')

try:
    from losses620 import *
    print('✓ LOSSES OK (with src in path)')
except Exception as e:
    print(f'✗ LOSSES FAILED: {e}')
PYEOF

# Test 2: Check if original run.py works (it should)
echo ""
echo "Test 2: Testing run module..."
python3 -c "import run; print('✓ RUN OK')" 2>&1

# Test 3: Validate style_families has the required function
echo ""
echo "Test 3: Checking validate_phase616_clean_contract..."
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/xy/Latent_Style/SchrodingerBridge/src')
from style_families import validate_phase616_clean_contract
print(f'✓ validate_phase616_clean_contract exists: {callable(validate_phase616_clean_contract)}')
PYEOF

echo ""
echo "=== Tests Complete ==="