#!/bin/bash
echo "=== Fixing src/__init__.py ==="
cat > /home/xy/Latent_Style/SchrodingerBridge/src/__init__.py << 'EOF'
from src.model import TimeConditionedLANCETBridge, build_model_from_config

__all__ = [
    "TimeConditionedLANCETBridge",
    "build_model_from_config",
]
EOF

echo "Updated __init__.py:"
cat /home/xy/Latent_Style/SchrodingerBridge/src/__init__.py
echo ""
echo "=== Re-running Import Test ==="
cd /home/xy/Latent_Style/SchrodingerBridge

python3 -c "from src.config_schema import *; print('✓ CONFIG_SCHEMA OK')" 2>&1
python3 -c "from src.model620 import *; print('✓ MODEL OK')" 2>&1
python3 -c "from src.losses620 import *; print('✓ LOSSES OK')" 2>&1
python3 -c "import run; print('✓ RUN OK')" 2>&1

echo ""
echo "=== All Import Tests Complete ==="