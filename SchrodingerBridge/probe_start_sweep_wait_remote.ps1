$ErrorActionPreference = "Stop"
$Root = "I:\Github\Latent_Style\SchrodingerBridge"
$OutLog = Join-Path $Root "logs\diffeomorphic_tangent_sweep_wait.out.log"
$ErrLog = Join-Path $Root "logs\diffeomorphic_tangent_sweep_wait.err.log"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
$p = Start-Process `
    -FilePath "python" `
    -ArgumentList @("tools\experiments\run_diffeomorphic_tangent_sweep.py", "--dry-run", "--max-experiments", "1") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -Wait `
    -PassThru
Write-Output ("sweep dry-run exit={0}" -f $p.ExitCode)
