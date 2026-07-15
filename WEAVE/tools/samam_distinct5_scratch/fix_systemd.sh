#!/usr/bin/env bash
# Fix: disable broken systemd in WSL to resolve user session timeout
# This is the root cause of training dying at step 2
set -e

echo "=== BEFORE ==="
cat /etc/wsl.conf

echo ""
echo "=== DISABLING SYSTEMD ==="
# Backup original
cp /etc/wsl.conf /etc/wsl.conf.bak.$(date +%s) 2>/dev/null || true

# Write new config without systemd
cat > /etc/wsl.conf << 'EOF'
[user]
default=xy

[boot]
systemd=false
EOF

echo "=== AFTER ==="
cat /etc/wsl.conf
echo ""
echo "=== DONE - need wsl --shutdown to apply ==="
