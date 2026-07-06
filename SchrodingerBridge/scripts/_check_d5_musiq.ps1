# Check D5 MUSIQ results and launch SaMam W20 v2 via ps1 wrapper
$ErrorActionPreference = "Continue"

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Host "=== D5 MUSIQ log ==="
$log = "$REPO\logs\eval_d5_musiq.log"
if (Test-Path $log) {
    Get-Content $log
}

Write-Host ""
Write-Host "=== D5 MUSIQ JSON results ==="
Get-ChildItem "$REPO\exp" -Filter "_eval_*_musiq.json" -ErrorAction SilentlyContinue |
    ForEach-Object {
        $r = Get-Content $_.FullName -Raw | ConvertFrom-Json
        Write-Host ("{0}: MUSIQ={1}" -f $_.Name, $r.musiq)
    }

Write-Host ""
Write-Host "=== VRAM ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
