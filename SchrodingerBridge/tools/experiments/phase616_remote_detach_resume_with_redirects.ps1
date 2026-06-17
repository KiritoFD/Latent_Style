$runScript = '/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/experiments/phase616_resume_training_state_foreground.sh'
$runDir = '/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical/h0_vertical_fm'
$stdoutPath = 'C:\Users\Administrator\phase616_resume_stdout.log'
$stderrPath = 'C:\Users\Administrator\phase616_resume_stderr.log'

Remove-Item $stdoutPath -ErrorAction SilentlyContinue
Remove-Item $stderrPath -ErrorAction SilentlyContinue

$proc = Start-Process -FilePath 'wsl.exe' `
    -ArgumentList 'bash', $runScript, $runDir `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

Start-Sleep -Seconds 5
if ($proc.HasExited) {
    Write-Output ("EXITED " + $proc.ExitCode)
} else {
    Write-Output ("RUNNING " + $proc.Id)
}
Write-Output ("STDOUT " + $stdoutPath)
Write-Output ("STDERR " + $stderrPath)
