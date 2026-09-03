$ErrorActionPreference = "Stop"

$repo = "I:\Github\Latent_Style\SchrodingerBridge"
$task = "LANCET_phase_envelope_full"
$out = Join-Path $repo "exp\vae_backend\ema_phase_envelope"
New-Item -ItemType Directory -Force -Path $out | Out-Null

cmd /c "schtasks /End /TN $task" | Out-Null
cmd /c "schtasks /Delete /TN $task /F" | Out-Null

$taskCmd = "cmd /c cd /d `"$repo`" && run_ema_phase_envelope_full_remote.cmd > exp\vae_backend\ema_phase_envelope\task_stdout.log 2> exp\vae_backend\ema_phase_envelope\task_stderr.log"
cmd /c "schtasks /Create /TN $task /TR `"$taskCmd`" /SC ONCE /ST 23:59 /F"
cmd /c "schtasks /Run /TN $task"
cmd /c "schtasks /Query /TN $task /V /FO LIST"
