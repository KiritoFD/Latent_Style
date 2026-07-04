#!/usr/bin/env bash
echo "=== all python in PATH ==="
ls /usr/bin/python* 2>/dev/null
ls /usr/local/bin/python* 2>/dev/null
echo "=== miniconda/anaconda ==="
ls /root/miniconda3/bin/python 2>/dev/null && echo "miniconda3 found"
ls /opt/conda/bin/python 2>/dev/null && echo "opt/conda found"
ls /home/*/miniconda3/bin/python 2>/dev/null
ls /home/*/anaconda3/bin/python 2>/dev/null
echo "=== which commands ==="
which python3 2>/dev/null
which python3.10 2>/dev/null
which python3.11 2>/dev/null
which python3.12 2>/dev/null
echo "=== mamba_ssm search ==="
find / -name "mamba_ssm" -type d 2>/dev/null | head -5
echo "=== pip list conda envs ==="
ls /root/.conda/envs/ 2>/dev/null
ls /home/*/.conda/envs/ 2>/dev/null
echo "=== shell init files ==="
cat /root/.bashrc 2>/dev/null | grep -E "conda|python|venv|PATH" | head -10
echo "=== DONE ==="
