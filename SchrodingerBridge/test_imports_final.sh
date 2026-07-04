#!/bin/bash
echo "=== Testing Imports from Project Root ==="
cd /home/xy/Latent_Style/SchrodingerBridge

# Add project root to PYTHONPATH
export PYTHONPATH="/home/xy/Latent_Style/SchrodingerBridge:$PYTHONPATH"

echo "Testing config_schema..."
python3 -c "import sys; sys.path.insert(0, '.'); from src.config_schema import *; print('✓ CONFIG_SCHEMA OK')" 2>&1

echo "Testing model620..."
python3 -c "import sys; sys.path.insert(0, '.'); from src.model620 import *; print('✓ MODEL OK')" 2>&1

echo "Testing losses620..."
python3 -c "import sys; sys.path.insert(0, '.'); from src.losses620 import *; print('✓ LOSSES OK')" 2>&1

echo "Testing style_families (checking for validate_phase616_clean_contract)..."
python3 -c "import sys; sys.path.insert(0, '.'); from src.style_families import validate_phase616_clean_contract; print('✓ style_families OK - has validate_phase616_clean_contract')" 2>&1

echo "Testing run module..."
python3 -c "import run; print('✓ RUN OK')" 2>&1

echo ""
echo "=== All Import Tests Complete ==="