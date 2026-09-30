#!/bin/bash
# Check if source files need syncing
set -euo pipefail

SRC_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge/src"

echo "=== Remote file hashes ==="
for f in losses620.py model620.py config_schema.py; do
    echo -n "$f: "
    md5sum "$SRC_DIR/$f" 2>/dev/null || echo "NOT FOUND"
done