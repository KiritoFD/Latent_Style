param(
    [Parameter(Mandatory = $true)]
    [string]$ConfigName
)

$ErrorActionPreference = "Stop"
$root = "I:\Github\Latent_Style\SchrodingerBridge"
$name = [IO.Path]::GetFileNameWithoutExtension($ConfigName)
$output = Join-Path $root "exp\$name"
New-Item -ItemType Directory -Force $output | Out-Null

$process = Start-Process `
    -FilePath "python" `
    -ArgumentList @("-u", "src\run.py", "--config", "configs\$ConfigName") `
    -WorkingDirectory $root `
    -RedirectStandardOutput (Join-Path $output "run.log") `
    -RedirectStandardError (Join-Path $output "err.log") `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path (Join-Path $output "pid.txt") -Value $process.Id -Encoding ascii
Write-Output $process.Id
