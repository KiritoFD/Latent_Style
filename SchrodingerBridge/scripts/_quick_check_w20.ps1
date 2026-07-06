# Quick check W20 gen status
$ErrorActionPreference = "Continue"
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Host "=== VRAM ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

Write-Host ""
Write-Host "=== SDTurbo W20 log ==="
$log = "$REPO\logs\sdturbo_w20_full.log"
if (Test-Path $log) {
    Get-Content $log -Tail 20
} else {
    Write-Host "no log yet"
}

Write-Host ""
Write-Host "=== SDTurbo images count ==="
$dir = "$REPO\exp\baseline_wikiarts20\sdturbo\images"
if (Test-Path $dir) {
    $cnt = (Get-ChildItem $dir -File).Count
    Write-Host "sdturbo: $cnt files"
}

Write-Host ""
Write-Host "=== Process ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select-Object ProcessId, @{N='CPU';E={$_.UserModeTime/1e7}}, @{N='Mem_MB';E={[math]::Round($_.WorkingSetSize/1MB)}} |
    Format-Table -Auto
