#!/bin/bash
echo "===FULL LOG==="
cat /mnt/i/exp_256_photo2art/_train_latent256_photo2art.log
echo ""
echo "===NOHUP OUT==="
cat /mnt/i/exp_256_photo2art/_train_latent256_photo2art.nohup 2>/dev/null || echo "NO NOHUP FILE"
echo "===LAUNCH SCRIPT==="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/launch_train_latent256.sh 2>/dev/null
echo "===DMESG OOM==="
dmesg 2>/dev/null | tail -20 | grep -iE "oom|killed|memory" || echo "(no dmesg access)"
