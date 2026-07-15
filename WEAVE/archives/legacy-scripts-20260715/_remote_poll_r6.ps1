$Remote = "administrator@100.115.18.62"
$LogPath = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs\launch.log"
$ErrPath = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs\launch.err"
$TrainLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs\train.log"

$sshBase = "ssh.exe -p 2222 -o LogLevel=ERROR $Remote"

function Invoke-RemoteCommand($cmd) {
    $full = "$sshBase `"$cmd`""
    Write-Host "> $full"
    Invoke-Expression $full
}

Write-Host "=== Remote process check ==="
Invoke-RemoteCommand "powershell -Command `"Get-Process powershell | Where-Object {`$_.CommandLine -like '*pipeline_probe_713_round6*'} | Select-Object Id,StartTime | Format-Table -AutoSize`""

Write-Host "=== launch.log tail ==="
Invoke-RemoteCommand "powershell -Command `"if (Test-Path '$LogPath') { Get-Content '$LogPath' -Tail 10 } else { Write-Host 'launch.log not found' }`""

Write-Host "=== launch.err tail ==="
Invoke-RemoteCommand "powershell -Command `"if (Test-Path '$ErrPath') { Get-Content '$ErrPath' -Tail 10 } else { Write-Host 'launch.err not found' }`""

Write-Host "=== train.log tail ==="
Invoke-RemoteCommand "powershell -Command `"if (Test-Path '$TrainLog') { Get-Content '$TrainLog' -Tail 10 } else { Write-Host 'train.log not found' }`""
