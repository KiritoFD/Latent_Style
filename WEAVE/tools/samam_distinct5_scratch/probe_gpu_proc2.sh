#!/usr/bin/env bash
echo "===== /tmp/wsl_keepalive.sh content ====="
cat /tmp/wsl_keepalive.sh 2>/dev/null

echo
echo "===== all wsl sessions (who) ====="
who 2>/dev/null
echo "---"
w 2>/dev/null | head -10

echo
echo "===== bash history tail (recent train/eval/keepalive invocations) ====="
tail -50 /home/xy/.bash_history 2>/dev/null | grep -iE 'samam|keepalive|train|eval|tmux|setsid|nohup' | tail -30

echo
echo "===== look for any auto-relaunch scripts (watch loops) ====="
ps -eo pid,ppid,user,etime,cmd --sort=-etime 2>/dev/null | grep -iE 'watch|loop|while|keepalive|relaunch' | grep -v grep | head -20

echo
echo "===== any backgrounded jobs in /home/xy ====="
ls -la /home/xy/*.sh /home/xy/*.log /home/xy/nohup.out 2>/dev/null | head -20

echo
echo "===== check SchrodingerBridge tools for self-relaunch scripts ====="
grep -rlE 'while true|relaunch|respawn|keepalive' /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/*.sh 2>/dev/null | head -10

echo
echo "===== content of remote_launch_eval.sh (suspect: keepalive tmux) ====="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_launch_eval.sh 2>/dev/null

echo
echo "===== content of quick_status.sh / quick_eval_status.sh ====="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/quick_eval_status.sh 2>/dev/null
