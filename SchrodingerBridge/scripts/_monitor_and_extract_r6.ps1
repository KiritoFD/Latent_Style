$Remote = "administrator@100.115.18.62"
$TrainLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\logs\train.log"
$EvalDone = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\full_eval\adain15\summary.json"
$DinoDone = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\full_eval\adain15\dino.json"

function Invoke-RemoteEncoded($psCmd) {
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($psCmd))
    $expr = "ssh.exe -p 2222 -o LogLevel=ERROR $Remote `"powershell -EncodedCommand $encoded`""
    return Invoke-Expression $expr
}

Write-Host "Monitoring Round 6 remote training..."
for ($i = 0; $i -lt 120; $i++) {
    $tail = Invoke-RemoteEncoded "if (Test-Path '$TrainLog') { Get-Content '$TrainLog' -Tail 3 } else { Write-Host 'train.log not found' }"
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $tail"
    $evalExists = Invoke-RemoteEncoded "Test-Path '$EvalDone'"
    $dinoExists = Invoke-RemoteEncoded "Test-Path '$DinoDone'"
    if (($tail -match "COMPLETE") -or (($evalExists -match "True") -and ($dinoExists -match "True"))) {
        Write-Host "Training/eval appears complete. Extracting results..."
        break
    }
    Start-Sleep -Seconds 60
}

Write-Host "=== Final extraction ==="
$extractCmd = "python I:\Github\Latent_Style\SchrodingerBridge\scripts\_extract_round6_results.py"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($extractCmd))
$expr = "ssh.exe -p 2222 -o LogLevel=ERROR $Remote `"powershell -EncodedCommand $encoded`""
Invoke-Expression $expr
