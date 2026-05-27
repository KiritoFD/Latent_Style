$task = 'T01Patch36'
$tr = 'cmd.exe /c "I:\Github\Latent_Style\SchrodingerBridge\run_t01_patch36_ssh.cmd"'
& schtasks.exe /Delete /TN $task /F
& schtasks.exe /Create /TN $task /TR $tr /SC ONCE /ST 23:59 /F
& schtasks.exe /Run /TN $task
Start-Sleep -Seconds 5
& schtasks.exe /Query /TN $task /FO LIST /V
