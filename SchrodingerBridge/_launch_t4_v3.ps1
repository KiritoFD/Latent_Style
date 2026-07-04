# Launch T4 queue v3 detached
$queue = "I:/Github/Latent_Style/SchrodingerBridge/_run_t4_queue_v3.ps1"
$launcherLog = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/_t4_launcher_v3.log"
"Launcher v3 start $(Get-Date)" | Out-File $launcherLog -Encoding utf8
$proc = Start-Process powershell -ArgumentList @('-ExecutionPolicy','Bypass','-NoProfile','-File',$queue) -WindowStyle Hidden -PassThru
"Launched PID=$($proc.Id) at $(Get-Date)" | Out-File $launcherLog -Append -Encoding utf8
Start-Sleep -Seconds 3
"After 3s, PID=$($proc.Id) HasExited=$($proc.HasExited)" | Out-File $launcherLog -Append -Encoding utf8
"Launcher v3 end $(Get-Date)" | Out-File $launcherLog -Append -Encoding utf8
