# Infra optimization: training benchmark — torch.compile + batch=48 + channels_last
# Usage: powershell -File _run_infra_train.ps1
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$exp = "infra_train_opt"
$config = "configs\$exp.json"
$logOut = "C:\Users\Administrator\logs\${exp}_train.out"

Write-Output "=== INFRA TRAIN OPT START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Output "Config: torch.compile(reduce-overhead) + batch=48 + channels_last + no gpu_monitor"
python -u src\run.py --config $config 2>&1 | Tee-Object -FilePath $logOut
$trainEc = $LASTEXITCODE
Write-Output "=== INFRA TRAIN OPT DONE exit=$trainEc $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# Extract timing from CSV
Write-Output "=== TIMING SUMMARY ==="
$csv = Get-ChildItem "exp\$exp\logs\training_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($csv) {
    $lines = Get-Content $csv.FullName
    $header = $lines[0].Split(',')
    $hIdx = @{}
    for ($i = 0; $i -lt $header.Length; $i++) { $hIdx[$header[$i]] = $i }
    foreach ($line in $lines[1..($lines.Length - 1)]) {
        $cols = $line.Split(',')
        $epoch = $cols[$hIdx['epoch']]
        $epochTime = $cols[$hIdx['epoch_time_sec']]
        $samplesPerSec = $cols[$hIdx['samples_per_sec']]
        $vramPeak = $cols[$hIdx['cuda_peak_allocated_gb']]
        $vramReserved = $cols[$hIdx['cuda_peak_reserved_gb']]
        Write-Output "epoch=$epoch  epoch_time=${epochTime}s  samples/s=${samplesPerSec}  VRAM_alloc=${vramPeak}GB  VRAM_reserved=${vramReserved}GB"
    }
}
