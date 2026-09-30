$scriptPath = '/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/experiments/phase616_resume_keeper.sh'
$proc = Start-Process -FilePath 'wsl.exe' -ArgumentList 'bash', $scriptPath -PassThru
Start-Sleep -Seconds 5
if ($proc.HasExited) {
    Write-Output ("EXITED " + $proc.ExitCode)
} else {
    Write-Output ("RUNNING " + $proc.Id)
}
