#!/bin/bash
# Find training launch command
echo "=== Checking logs ==="
ls -la /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/logs/ 2>/dev/null
echo ""
echo "=== Checking run.py ==="
cat /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/src/run.py 2>/dev/null | grep -n "def main\|argparse\|if __name__" | head -10
echo ""
echo "=== Checking how training was launched ==="
# Look for a launch script or shell script
find /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/ -name "*.sh" -maxdepth 2 2>/dev/null | head -5
echo ""
# Check the main repo for a launch script
find /mnt/i/Github/Latent_Style/SchrodingerBridge/ -name "*.sh" -maxdepth 2 2>/dev/null | head -5