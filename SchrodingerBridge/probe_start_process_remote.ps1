$ErrorActionPreference = "Stop"
$Root = "I:\Github\Latent_Style\SchrodingerBridge"
$OutLog = Join-Path $Root "logs\sp_probe.out.log"
$ErrLog = Join-Path $Root "logs\sp_probe.err.log"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
$p = Start-Process `
    -FilePath "python" `
    -ArgumentList @("-c", "print(123)") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -Wait `
    -PassThru
Write-Output ("probe exit={0}" -f $p.ExitCode)
