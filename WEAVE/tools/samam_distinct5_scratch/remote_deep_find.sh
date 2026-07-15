#!/usr/bin/env bash
echo "=== whoami ==="
whoami
echo "=== home ==="
echo $HOME
ls -la $HOME/ 2>&1 | head -20
echo "=== home venvs ==="
ls $HOME/venvs/ 2>/dev/null
ls $HOME/.venv/ 2>/dev/null
ls $HOME/miniconda3/ 2>/dev/null
ls $HOME/anaconda3/ 2>/dev/null
echo "=== /home/xy ==="
ls /home/xy/ 2>/dev/null | head -20
echo "=== /mnt/i Python ==="
ls /mnt/i/ 2>/dev/null | grep -i -E "python|venv|conda" | head -5
echo "=== Windows Python venvs ==="
ls /mnt/c/Users/Administrator/ 2>/dev/null | grep -i -E "venv|conda|python|miniconda|anaconda" | head -10
ls /mnt/c/ProgramData/ 2>/dev/null | grep -i -E "conda|python" | head -5
echo "=== all mamba_ssm ==="
find / -maxdepth 8 -name "mamba_ssm*" 2>/dev/null | head -10
echo "=== all samam venvs ==="
find / -maxdepth 6 -name "samam" -type d 2>/dev/null | head -10
echo "=== pytorch_lightning versions ==="
find / -maxdepth 8 -name "pytorch_lightning" -type d 2>/dev/null | head -10
echo "=== DONE ==="
