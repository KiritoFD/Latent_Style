#!/bin/bash
echo "=== Full mamba_full.log ==="
cat /tmp/mamba_full.log
echo ""
echo "=== Find any ninja error logs ==="
find /tmp/mamba_full -name "*.log" -exec ls -la {} \;
find /tmp/mamba_full -name "ninja*.log" -exec cat {} \;
echo ""
echo "=== Check build.ninja ==="
find /tmp/mamba_full -name "build.ninja" -exec head -50 {} \;
echo ""
echo "=== Check /tmp/pip-* logs ==="
ls -la /tmp/pip-* 2>/dev/null | head -5
