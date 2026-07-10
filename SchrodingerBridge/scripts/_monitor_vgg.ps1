$pid_check = 19100
$p1 = Get-Process -Id $pid_check -ErrorAction SilentlyContinue
if (-not $p1) {
    Write-Host "Process $pid_check not found - may have finished or crashed"
    exit 0
}
$cpu1 = $p1.CPU
$ws1 = $p1.WS
# Check GPU
$gpu1 = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null
Write-Host "T0: CPU=${cpu1}s WS=$([math]::Round($ws1/1MB,0))MB GPU=${gpu1}"
Start-Sleep -Seconds 10
$p2 = Get-Process -Id $pid_check -ErrorAction SilentlyContinue
if (-not $p2) {
    Write-Host "Process died during wait"
    exit 0
}
$cpu2 = $p2.CPU
$ws2 = $p2.WS
$gpu2 = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null
Write-Host "T1: CPU=${cpu2}s WS=$([math]::Round($ws2/1MB,0))MB GPU=${gpu2}"
Write-Host "Delta CPU=$([math]::Round($cpu2-$cpu1,1))s in 10s"
# Check images count
$imgDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_weave\full_eval_s1\s1_vgg_opt_wct_ll05\images"
$imgCount = (Get-ChildItem $imgDir -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "Images count: $imgCount"
# Check for any metrics files
$baseDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_weave\full_eval_s1\s1_vgg_opt_wct_ll05"
Get-ChildItem $baseDir -Recurse -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  File: $($_.FullName) Size=$($_.Length)" }
