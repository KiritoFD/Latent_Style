function Invoke-RemoteEncoded($psCmd) {
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($psCmd))
    $expr = "ssh.exe -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 `"powershell -EncodedCommand $encoded`""
    return Invoke-Expression $expr
}

# Check if results exist
$evalPath = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\full_eval\adain15\summary.json"
$dinoPath = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\full_eval\adain15\dino.json"

Write-Host "=== Check eval files ==="
Invoke-RemoteEncoded "Write-Host 'eval=' (Test-Path '$evalPath') 'dino=' (Test-Path '$dinoPath')"

Write-Host "=== Train log tail ==="
Invoke-RemoteEncoded "if (Test-Path 'I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs\train.log') { Get-Content 'I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs\train.log' -Tail 10 } else { Write-Host 'train.log not found' }"

Write-Host "=== Eval log tail ==="
Invoke-RemoteEncoded "if (Test-Path 'I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs\eval_adain15.log') { Get-Content 'I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs\eval_adain15.log' -Tail 10 } else { Write-Host 'eval_adain15.log not found' }"

Write-Host "=== DINO log tail ==="
Invoke-RemoteEncoded "if (Test-Path 'I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs\dino.log') { Get-Content 'I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs\dino.log' -Tail 10 } else { Write-Host 'dino.log not found' }"

Write-Host "=== Extract results ==="
Invoke-RemoteEncoded "python I:\Github\Latent_Style\SchrodingerBridge\scripts\_extract_round6_results.py"
