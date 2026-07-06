# Stop W20 gen and estimate cost
$ErrorActionPreference = "Continue"

Write-Host "=== Stopping W20 gen ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe'" |
    Where-Object {
        $_.CommandLine -like "*_w20_full_gen*" -or
        $_.CommandLine -like "*_gen_sdturbo_w20*" -or
        $_.CommandLine -like "*_gen_samam_w20*"
    } |
    ForEach-Object {
        Write-Host "Killing PID $($_.ProcessId) ($($_.Name))"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

schtasks /End /TN "w20_full_gen" 2>$null
schtasks /Delete /TN "w20_full_gen" /F 2>$null

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "=== VRAM after cleanup ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

Write-Host ""
Write-Host "=== Quick benchmark: SDTurbo single image time ==="
# Run a quick 5-image benchmark to estimate
$PYTHON = "C:\Program Files\Python312\python.exe"
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:HF_HOME = "C:\Users\Administrator\.cache\huggingface"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TORCH_HOME = "C:\Users\Administrator\.cache\torch"
$env:PYTHONPATH = "$REPO\src;$USER_SITE;$REPO\scripts"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"

# Check sdturbo log to see rate
$log = "$REPO\logs\sdturbo_w20_full.log"
if (Test-Path $log) {
    Write-Host "=== SDTurbo log ==="
    Get-Content $log -Tail 30
}

Write-Host ""
Write-Host "=== Existing W20 image counts ==="
$base = "$REPO\exp\baseline_wikiarts20"
$methods = @("sdturbo", "styleid", "samst", "samam", "cut")
foreach ($m in $methods) {
    $dir = "$base\$m\images"
    if (Test-Path $dir) {
        $cnt = (Get-ChildItem $dir -File).Count
        Write-Host "  ${m}: $cnt / 12000"
    }
}
