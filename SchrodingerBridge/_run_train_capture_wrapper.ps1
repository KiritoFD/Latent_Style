param([string]$ConfigName, [string]$ExpName)
Set-Location I:\Github\Latent_Style\SchrodingerBridge
$env:PYTHONPATH = ""
if (-not $ExpName) { $ExpName = $ConfigName }
$errLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\${ExpName}_stderr.log"
$outLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\${ExpName}_stdout.log"

# Start-Process with -RedirectStandardError creates a child process WITHOUT console
# This avoids fortrl error (200) while still capturing stderr/stdout
# -Wait keeps the wrapper alive until python exits (so schtask tracks it correctly)
$proc = Start-Process -FilePath "C:\Program Files\Python312\python.exe" `
    -ArgumentList "run.py --config configs/${ConfigName}.json" `
    -WorkingDirectory "I:\Github\Latent_Style\SchrodingerBridge" `
    -RedirectStandardError $errLog `
    -RedirectStandardOutput $outLog `
    -WindowStyle Hidden `
    -Wait `
    -PassThru

Write-Output "PYTHON_EXIT_CODE: $($proc.ExitCode)"
