#!/bin/bash
echo "=== error/fatal/ninja lines from build log ==="
grep -iE "error:|fatal|undefined|cannot|Failed|ninja" /tmp/mamba_build.log | head -50
echo ""
echo "=== Lines around 'ninja' ==="
grep -n "ninja" /tmp/mamba_build.log | head -20
echo ""
echo "=== Lines 200-300 of log ==="
sed -n '200,300p' /tmp/mamba_build.log
