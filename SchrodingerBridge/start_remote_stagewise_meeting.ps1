$ErrorActionPreference = "Stop"

$Remote = "administrator@100.115.18.62"
$RemoteCommand = 'I:\Github\Latent_Style\SchrodingerBridge\run_stagewise_meeting_ssh.cmd'
$Args = @(
    "-p", "2222",
    "-o", "LogLevel=ERROR",
    $Remote,
    $RemoteCommand
)

$process = Start-Process `
    -FilePath "ssh.exe" `
    -ArgumentList $Args `
    -WindowStyle Hidden `
    -PassThru

Write-Output ("Started remote stagewise meeting over persistent SSH: local ssh PID={0}" -f $process.Id)
