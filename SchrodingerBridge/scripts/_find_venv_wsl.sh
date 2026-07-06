#!/bin/bash
# Find python venvs in WSL
echo "=== Home dirs ==="
ls /home/ 2>&1

echo ""
echo "=== Common venv locations ==="
ls /home/*/venv/bin/python 2>/dev/null
ls /home/*/.venv/bin/python 2>/dev/null
ls /home/*/env/bin/python 2>/dev/null
ls /venv/bin/python 2>/dev/null
ls /env/bin/python 2>/dev/null

echo ""
echo "=== Find activate scripts ==="
find / -name "activate" -path "*/bin/*" 2>/dev/null | head -10

echo ""
echo "=== Find python binaries ==="
find / -name "python3.*" -type f -executable 2>/dev/null | head -10

echo ""
echo "=== pip3 list ==="
pip3 list 2>/dev/null | head -20

echo ""
echo "=== apt python3 ==="
dpkg -l | grep -i python3 | head -10

echo ""
echo "=== Look for SchrodingerBridge venv ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/venv 2>&1
ls /mnt/i/Github/Latent_Style/venv 2>&1
ls /mnt/i/Github/Latent_Style/.venv 2>&1
