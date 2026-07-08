#!/bin/bash
# Install mamba-ssm 1.2.2 with full output captured to log
source /root/samam_venv/bin/activate

LOG=/tmp/mamba_install.log
echo "=== Installing mamba-ssm 1.2.2, logging to $LOG ==="
pip install mamba-ssm==1.2.2 --no-build-isolation -v > $LOG 2>&1
EXIT=$?
echo "=== pip exit code: $EXIT ==="
echo ""
echo "=== LAST 80 lines of log ==="
tail -80 $LOG
echo ""
echo "=== ERROR lines from log ==="
grep -iE "error|fatal|undefined" $LOG | tail -30
