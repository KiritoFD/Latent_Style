$cmd = @"
Start-Process -FilePath powershell.exe -ArgumentList '-ExecutionPolicy Bypass -File "I:\Github\Latent_Style\SchrodingerBridge\scripts\_pipeline_probe_713_round6_scratch10.ps1"' -WindowStyle Hidden -RedirectStandardOutput 'I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs\launch.log' -RedirectStandardError 'I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs\launch.err'
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))
$expr = "ssh.exe -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 `"powershell -EncodedCommand $encoded`""
Write-Host "> $expr"
Invoke-Expression $expr
