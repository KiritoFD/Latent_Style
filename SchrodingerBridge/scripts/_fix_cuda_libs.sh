#!/bin/bash
# Kill current rebuild, install cuda-libraries-dev-12-1 properly, then rebuild

echo "=== STEP 1: Kill current pip/ninja ==="
pkill -9 -f "pip install" 2>/dev/null || true
pkill -9 -f "ninja" 2>/dev/null || true
pkill -9 -f "nvcc" 2>/dev/null || true
pkill -9 -f "cc1plus" 2>/dev/null || true
pkill -9 -f "cicc" 2>/dev/null || true
pkill -9 -f "ptxas" 2>/dev/null || true
sleep 2
ps -ef | grep -E "pip|ninja|nvcc" | grep -v grep | head -5

echo ""
echo "=== STEP 2: Check apt sources ==="
ls /etc/apt/sources.list.d/ | head -10
cat /etc/apt/sources.list.d/*.list 2>/dev/null | head -20

echo ""
echo "=== STEP 3: Try installing via cuda-toolkit ==="
# First try: maybe apt-key was missing
apt-get update -qq 2>&1 | tail -5

# Try various package name variants
echo "--- Trying cuda-libraries-dev-12-1 ---"
apt-get install -y --no-install-recommends cuda-libraries-dev-12-1 2>&1 | tail -10

echo ""
echo "--- Trying libcusparse-dev (if available) ---"
apt-cache search cusparse 2>&1 | head -5
apt-cache search cublas 2>&1 | head -5

echo ""
echo "=== STEP 4: If still missing, download cusparse.h manually ==="
ls /usr/local/cuda/include/cusparse.h 2>&1 || true

# If apt didn't work, download the .deb manually from Ubuntu repos
if [ ! -f /usr/local/cuda/include/cusparse.h ]; then
    echo "cusparse.h still missing, trying manual deb download..."
    # The cuda-libraries-dev-12-1 package from NVIDIA's apt repo
    # Try adding nvidia repo
    apt-get install -y wget gnupg 2>&1 | tail -3
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb 2>&1
    dpkg -i /tmp/cuda-keyring.deb 2>&1 | tail -5
    apt-get update -qq 2>&1 | tail -5
    apt-get install -y --no-install-recommends cuda-libraries-dev-12-1 2>&1 | tail -10
fi

echo ""
echo "=== STEP 5: Verify cusparse.h ==="
find /usr -name "cusparse.h" 2>/dev/null | head -5
ls -la /usr/local/cuda/include/cusparse.h 2>&1 || true

echo ""
echo "DONE_PHASE1=$(date)"
