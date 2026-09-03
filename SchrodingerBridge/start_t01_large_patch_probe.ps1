$task = 'T01LargePatchProbe'
$tr = 'cmd.exe /c "I:\Github\Latent_Style\SchrodingerBridge\run_t01_large_patch_probe_ssh.cmd"'
& schtasks.exe /Delete /TN $task /F
& schtasks.exe /Create /TN $task /TR $tr /SC ONCE /ST 23:59 /F
& schtasks.exe /Run /TN $task
Start-Sleep -Seconds 4
& schtasks.exe /Query /TN $task /FO LIST /V
