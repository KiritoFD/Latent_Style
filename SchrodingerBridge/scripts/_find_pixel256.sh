#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/i/Github/Latent_Style/SchrodingerBridge
echo "===pixel256 dirs==="
find "$ROOT/exp" -maxdepth 3 -type d -name "*pixel*" 2>/dev/null
echo "===pixel256 ckpts==="
find "$ROOT/exp" -maxdepth 4 -name "*.pt" -path "*pixel*" 2>/dev/null
echo "===root exp dirs==="
ls -la "$ROOT/exp" 2>/dev/null | head -30
