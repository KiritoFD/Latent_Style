#!/usr/bin/env bash
echo "=== /root/venvs ==="
ls /root/venvs/ 2>/dev/null || echo "no /root/venvs"
echo "=== /root/venvs/samam ==="
ls /root/venvs/samam/bin/ 2>/dev/null | head -5
echo "=== /root/.bashrc | grep venv ==="
grep -E "venv|conda|python|PATH" /root/.bashrc 2>/dev/null | head -10
echo "=== /home ==="
ls /home/ 2>/dev/null
echo "=== all venvs in / ==="
find / -maxdepth 4 -name "activate" -path "*/bin/*" 2>/dev/null | head -10
echo "=== mamba_ssm in any site-packages ==="
find / -maxdepth 7 -name "mamba_ssm" -type d 2>/dev/null | head -5
echo "=== conda envs ==="
find / -maxdepth 4 -name "conda" -type f 2>/dev/null | head -5
echo "=== DONE ==="
