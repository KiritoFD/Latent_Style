param([string]$ConfigName, [string]$ExpName)
Set-Location I:\Github\Latent_Style\SchrodingerBridge
$env:PYTHONPATH = ""
# Fix fortrl error (200): program aborting due to window-CLOSE event
# Intel Fortran runtime (via numpy/MKL) aborts when console closes in non-interactive schtask session
$env:FOR_DISABLE_CONSOLE = "1"
if (-not $ExpName) { $ExpName = $ConfigName }
$logFile = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\${ExpName}_train.log"
& "C:\Program Files\Python312\python.exe" run.py --config "configs/${ConfigName}.json" *> $logFile
