$ErrorActionPreference = 'Continue'
Set-Location 'I:\GitHub\Latent_Style\SchrodingerBridge'
$py = 'C:\Program Files\Python312\python.exe'
$args = @('-u', 'tools\remote_master_baseline_v2.py')
$out = 'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\master_stdout.log'
$err = 'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\master_stderr.log'
& $py @args > $out 2> $err
