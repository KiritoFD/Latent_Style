$runScript = '/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/experiments/phase616_resume_training_state_foreground.sh'
$runDir = '/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical/h0_vertical_fm'

$proc = Start-Process -FilePath 'wsl.exe' -ArgumentList 'bash', $runScript, $runDir -PassThru
Start-Sleep -Seconds 5
if ($proc.HasExited) {
    Write-Output ("EXITED " + $proc.ExitCode)
} else {
    Write-Output ("RUNNING " + $proc.Id)
}
