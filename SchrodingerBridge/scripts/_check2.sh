#!/bin/bash
echo "=== build dir status ==="
ls -la /tmp/mamba_full/mamba-1.2.2/build/temp.linux-x86_64-3.10/csrc/selective_scan/ 2>/dev/null | head -20
echo ""
echo "=== env vars of current pip (PID 1954) ==="
cat /proc/1954/environ 2>/dev/null | tr '\0' '\n' | grep -iE "mamba|force|build" | head -5
echo ""
echo "=== ps tree ==="
pstree -p 1954 2>/dev/null | head -10
echo ""
echo "=== check if /tmp has new install log ==="
ls -la /tmp/*.log 2>/dev/null | tail -5
echo ""
echo "=== latest nvcc output (file /proc/PID/fd/1) ==="
ls -la /proc/1999/fd/1 2>/dev/null
ls -la /proc/1999/fd/2 2>/dev/null
echo ""
echo "=== check if mamba_ssm built (site-packages) ==="
ls /root/samam_venv/lib/python3.10/site-packages/ | grep -i mamba 2>/dev/null
echo ""
echo "=== free VRAM ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null
