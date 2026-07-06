# Stop Windows SaMam immediately
$ErrorActionPreference = "Continue"

Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe'" |
    Where-Object {
        $_.CommandLine -like "*_gen_samam*" -or
        $_.CommandLine -like "*samam_random5*"
    } |
    ForEach-Object {
        Write-Host "Killing PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

schtasks /End /TN "samam_random5" 2>$null
schtasks /Delete /TN "samam_random5" /F 2>$null

Start-Sleep -Seconds 2

Write-Host "=== VRAM ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

Write-Host ""
Write-Host "=== Check WSL mamba-ssm ==="
wsl -e bash -c "which python3 && python3 -c 'import mamba_ssm; print(\"mamba_ssm OK:\", mamba_ssm.__version__)' 2>&1"
wsl -e bash -c "python3 -c 'import torch; print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available())' 2>&1"

Write-Host ""
Write-Host "=== WSL SaMam repo ==="
wsl -e bash -c "ls -la /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/final_model.ckpt 2>&1"
wsl -e bash -c "ls /mnt/i/datasets/wikiarts20_512_test/ 2>&1 | head -5"

Write-Host ""
Write-Host "=== SaMam existing images ==="
$dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\samam\images"
if (Test-Path $dir) {
    $cnt = (Get-ChildItem $dir -File).Count
    Write-Host "samam: $cnt / 750"
}
