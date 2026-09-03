$ErrorActionPreference = "Stop"

$Root = "I:\Github\Latent_Style\SchrodingerBridge"
$Python = "python"
$Script = "tools\experiments\run_diffeomorphic_tangent_sweep.py"
$OutLog = Join-Path $Root "logs\diffeomorphic_tangent_sweep_sp.out.log"
$ErrLog = Join-Path $Root "logs\diffeomorphic_tangent_sweep_sp.err.log"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

$process = Start-Process `
    -FilePath $Python `
    -ArgumentList @($Script, "--force-train") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Output ("Started diffeomorphic tangent sweep: PID={0}" -f $process.Id)
