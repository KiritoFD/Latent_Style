param(
    [Parameter(Mandatory=$true)][string]$ExpName,
    [Parameter(Mandatory=$true)][string]$LowpassMode,
    [Parameter(Mandatory=$true)][string]$Alpha,
    [Parameter(Mandatory=$true)][string]$Kernel,
    [Parameter(Mandatory=$true)][string]$MbMode,
    [Parameter(Mandatory=$true)][string]$Triband,
    [Parameter(Mandatory=$true)][string]$MidScale,
    [Parameter(Mandatory=$true)][string]$HhScale
)

$ErrorActionPreference = "Continue"
Set-Location "I:/Github/Latent_Style/SchrodingerBridge"

$logDir = "exp/p4_fusion_breakout/infer_ablation"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$logFile = "$logDir/$ExpName.log"
$startTime = Get-Date
"[$startTime] START $ExpName lowpass=$LowpassMode alpha=$Alpha k=$Kernel mb=$MbMode triband=$Triband mid=$MidScale hh=$HhScale" | Out-File -FilePath $logFile -Encoding utf8

$pyExe = "C:\Program Files\Python312\python.exe"
if (-not (Test-Path $pyExe)) {
    $pyExe = (Get-Command python).Source
}
"Using python: $pyExe" | Out-File -FilePath $logFile -Append -Encoding utf8

& $pyExe _p4_infer_ablation.py $ExpName $LowpassMode $Alpha $Kernel $MbMode $Triband $MidScale $HhScale 2>&1 | Tee-Object -FilePath $logFile -Append

$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds
"[$endTime] END $ExpName duration=${duration}s" | Out-File -FilePath $logFile -Append -Encoding utf8
