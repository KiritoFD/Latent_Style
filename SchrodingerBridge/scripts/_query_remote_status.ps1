# Query remote pipeline status
$ErrorActionPreference = "Continue"

Write-Host "=== Python processes ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select-Object ProcessId, @{N='Start';E={$_.CreationDate}}, CommandLine |
    Format-List

Write-Host ""
Write-Host "=== samam_w20 progress ==="
$prog = "I:\Github\Latent_Style\SchrodingerBridge\exp\samam_w20_progress.json"
if (Test-Path $prog) {
    Get-Content $prog -Raw
} else {
    Write-Host "no progress file"
}

Write-Host ""
Write-Host "=== samam_w20 images count ==="
$imgDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\samam_w20\images"
if (Test-Path $imgDir) {
    $cnt = (Get-ChildItem $imgDir -File).Count
    Write-Host "images: $cnt"
} else {
    Write-Host "no images dir"
}

Write-Host ""
Write-Host "=== pipeline results ==="
$pipe = "I:\Github\Latent_Style\SchrodingerBridge\exp\_pipeline_fill_results.json"
if (Test-Path $pipe) {
    Get-Content $pipe -Raw
} else {
    Write-Host "no pipeline results file"
}

Write-Host ""
Write-Host "=== cut w20 eval ==="
$cut = "I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_cut_w20.json"
if (Test-Path $cut) {
    Get-Content $cut -Raw
} else {
    Write-Host "no cut w20 file"
}

Write-Host ""
Write-Host "=== scheduler tasks ==="
schtasks /Query /TN "*pipeline*" /FO LIST 2>$null
schtasks /Query /TN "*samam*" /FO LIST 2>$null
schtasks /Query /TN "*watchdog*" /FO LIST 2>$null
