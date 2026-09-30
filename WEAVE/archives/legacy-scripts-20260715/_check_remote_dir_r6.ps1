$cmd = "Test-Path 'I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs'"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))
$expr = "ssh.exe -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 `"powershell -EncodedCommand $encoded`""
Invoke-Expression $expr
