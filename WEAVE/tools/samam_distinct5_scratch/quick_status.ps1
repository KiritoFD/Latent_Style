# Quick status - extract only step number, no train.log dump
$step = &wsl -d Ubuntu-22.04 -e bash -c "tr '\r' '\n' < /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log 2>/dev/null | grep -oP 'Epoch 0:.*?\|\s+\K\d+' | tail -1"
$train_pid = &wsl -d Ubuntu-22.04 -e bash -c "pgrep -f train_SaMam | head -1"
$gpu = &wsl -d Ubuntu-22.04 -e bash -c "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
$ckpt = &wsl -d Ubuntu-22.04 -e bash -c "ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints/ 2>/dev/null | wc -l"
$mon = &wsl -d Ubuntu-22.04 -e bash -c "pgrep -f remote_loop_monitor 2>/dev/null | head -1"
$prog = &wsl -d Ubuntu-22.04 -e bash -c "tail -3 /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/progress.log 2>/dev/null"
$wsl_count = (Get-Process -Name wsl -ErrorAction SilentlyContinue | Measure-Object).Count

Write-Host "Step: $step / 7000"
Write-Host "Train PID: $train_pid"
Write-Host "GPU: $gpu"
Write-Host "Checkpoints: $ckpt"
Write-Host "Monitor PID: $mon"
Write-Host "WSL procs: $wsl_count"
Write-Host "--- Progress log ---"
Write-Host $prog
