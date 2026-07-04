#!/bin/bash
echo "=== Simplifying src/__init__.py to avoid circular imports ==="
cat > /home/xy/Latent_Style/SchrodingerBridge/src/__init__.py << 'EOF'
# Package initialization - imports deferred to avoid circular dependencies
EOF

echo "Updated __init__.py (empty)"
echo ""
echo "=== Re-running Import Test ==="
cd /home/xy/Latent_Style/SchrodingerBridge

echo "Testing config_schema..."
python3 -c "from src.config_schema import *; print('✓ CONFIG_SCHEMA OK')" 2>&1

echo "Testing model620..."
python3 -c "from src.model620 import *; print('✓ MODEL OK')" 2>&1

echo "Testing losses620..."
python3 -c "from src.losses620 import *; print('✓ LOSSES OK')" 2>&1

echo "Testing run..."
python3 -c "import run; print('✓ RUN OK')" 2>&1

echo ""
echo "=== Import Tests Complete ==="