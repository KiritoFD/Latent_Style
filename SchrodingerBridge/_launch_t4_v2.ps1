# Launch T4 queue detached - avoids SSH stream bottleneck
$queue = "I:/Github/Latent_Style/SchrodingerBridge/_run_t4_queue_v2.ps1"
$launcherLog = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/_t4_launcher.log"
"Launcher start $(Get-Date)" | Out-File $launcherLog -Encoding utf8
Start-Process powershell -ArgumentList @('-ExecutionPolicy','Bypass','-NoProfile','-File',$queue) -WindowStyle Hidden -PassThru | Select-Object Id,ProcessName | Out-File $launcherLog -Append -Encoding utf8
"Launcher end $(Get-Date)" | Out-File $launcherLog -Append -Encoding utf8
