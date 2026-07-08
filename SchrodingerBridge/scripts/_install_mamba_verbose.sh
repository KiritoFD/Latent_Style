#!/bin/bash
# Install mamba-ssm 1.2.2 with verbose error output
set -e
source /root/samam_venv/bin/activate

echo "=== Install mamba-ssm 1.2.2 (no-build-isolation, verbose) ==="
pip install mamba-ssm==1.2.2 --no-build-isolation -v 2>&1 | tail -100

echo ""
echo "=== DONE ==="
