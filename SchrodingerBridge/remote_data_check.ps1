# SSH到远程服务器获取真实实验数据
$sshHost = "administrator@100.115.18.62"
$sshPort = "2222"

Write-Host "=== 1. 检查SaMam HF评估数据 ===" -ForegroundColor Cyan
ssh -p $sshPort $sshHost "cat /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/curve_metrics_hf.csv | tail -20"

Write-Host "`n=== 2. 检查SaMam训练日志（获取训练时间） ===" -ForegroundColor Cyan
ssh -p $sshPort $sshHost "find /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch -name 'train.log' -exec grep -H 'Training completed' {} \;"

Write-Host "`n=== 3. 检查FC-SB T11训练日志（获取训练时间） ===" -ForegroundColor Cyan
ssh -p $sshPort $sshHost "grep -E 'Training completed|Total training time' /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/630_local_t11_stochastic_dwt_p08/train.log"

Write-Host "`n=== 4. 检查FC-SB推理时间 ===" -ForegroundColor Cyan
ssh -p $sshPort $sshHost "grep -E 'Inference time|Generation time|lancet_generation' /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/630_local_t11_stochastic_dwt_p08/full_eval/epoch_0005/summary.json"

Write-Host "`n=== 5. 检查unified_results.json ===" -ForegroundColor Cyan
ssh -p $sshPort $sshHost "cat /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/72/unified_results.json | head -100"
