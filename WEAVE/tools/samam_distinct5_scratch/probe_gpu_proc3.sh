#!/usr/bin/env bash
echo "===== remote_loop_monitor.sh ====="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_loop_monitor.sh 2>/dev/null

echo
echo "===== remote_launch_persistent.sh ====="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_launch_persistent.sh 2>/dev/null

echo
echo "===== remote_launch_only.sh ====="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_launch_only.sh 2>/dev/null

echo
echo "===== remote_progress_logger.sh ====="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_progress_logger.sh 2>/dev/null

echo
echo "===== start_monitor.sh ====="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/start_monitor.sh 2>/dev/null

echo
echo "===== remote_full_diag.sh ====="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_full_diag.sh 2>/dev/null

echo
echo "===== last 50 lines of keepalive.log ====="
tail -50 /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/keepalive.log 2>/dev/null

echo
echo "===== last 30 lines of eval.log ====="
tail -30 /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval.log 2>/dev/null

echo
echo "===== last 30 lines of train.log ====="
tail -30 /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log 2>/dev/null | tr '\r' '\n' | tail -30

echo
echo "===== look for any watchdog/relaunch loops still alive ====="
ps -ef 2>/dev/null | grep -iE 'remote_|monitor|launch|watchdog|loop' | grep -v grep | head -20

echo
echo "===== all bash -c processes ====="
ps -eo pid,ppid,user,etime,cmd 2>/dev/null | grep -E 'bash -c|bash /' | grep -v grep | head -20
