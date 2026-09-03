Set-Location 'I:\Github\Latent_Style\SchrodingerBridge'
$py = @"
import ast
from pathlib import Path
for p in [
    'src/config_schema.py',
    'src/lancet_backbone.py',
    'src/lancet_blocks.py',
    'src/lancet_runtime.py',
    'src/losses.py',
    'src/trainer.py',
    'src/utils/run_evaluation.py',
    'tools/experiments/run_vae_backend_256_probe.py',
]:
    ast.parse(Path(p).read_text(encoding='utf-8-sig'), filename=p)
    print('remote ast ok', p)
"@
& 'C:\Program Files\Python312\python.exe' -B -c $py
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

$task = 'LANCET_transport_texton_alloc'
try {
    Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue
} catch {}
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c I:\Github\Latent_Style\SchrodingerBridge\run_ema_transport_texton_alloc_remote.cmd'
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger -Force | Out-Null
Start-ScheduledTask -TaskName $task
Start-Sleep -Seconds 3
Get-ScheduledTask -TaskName $task | Select-Object TaskName,State | Format-List | Out-String -Width 200
if (Test-Path 'exp\vae_backend\ema_transport_texton_alloc\task.log') {
    Get-Content 'exp\vae_backend\ema_transport_texton_alloc\task.log' -Tail 20
}
