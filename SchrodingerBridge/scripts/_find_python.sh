#!/bin/bash
# Find python on remote
echo "=== which python3 ==="
which python3 2>&1
echo "=== which python ==="
which python 2>&1
echo "=== ls /usr/bin/python* ==="
ls /usr/bin/python* 2>&1
echo "=== ls ~/.local/bin/python* ==="
ls ~/.local/bin/python* 2>&1
echo "=== conda envs ==="
which conda 2>&1
conda env list 2>&1 | head -5
echo "=== pyenv ==="
which pyenv 2>&1
echo "=== PATH ==="
echo $PATH | tr ':' '\n' | head -20
