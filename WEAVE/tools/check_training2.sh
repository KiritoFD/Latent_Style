#!/bin/bash
echo "=== All python processes ==="
ps aux | grep python
echo ""
echo "=== Check for PID 13036 ==="
ps -p 13036 2>/dev/null || echo "PID 13036 not found"
echo ""
echo "=== Check nohup.out in src ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/src/nohup.out 2>/dev/null
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/src/nohup.out 2>/dev/null | tail -20
echo ""
echo "=== Check nohup.out in home ==="
ls -la ~/nohup.out 2>/dev/null
cat ~/nohup.out 2>/dev/null | tail -20
echo ""
echo "=== Check GPU ==="
nvidia-smi 2>/dev/null | head -20 || echo "nvidia-smi not available"