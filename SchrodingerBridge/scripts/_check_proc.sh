#!/bin/bash
echo "=== ps pip/python ==="
ps aux | grep -E "pip|python|torch" | grep -v grep | head -20
echo ""
echo "=== venv check ==="
ls -la /root/samam_venv/bin/ 2>/dev/null | head -20
echo ""
echo "=== pip list ==="
/root/samam_venv/bin/pip list 2>/dev/null | head -30
