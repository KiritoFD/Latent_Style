$Remote = "administrator@100.115.18.62"
$EvalDone = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\full_eval\adain15\summary.json"
$DinoDone = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_scratch_10ep\full_eval\adain15\dino.json"

function Invoke-RemoteEncoded($psCmd) {
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($psCmd))
    $expr = "ssh.exe -p 2222 -o LogLevel=ERROR $Remote `"powershell -EncodedCommand $encoded`""
    return Invoke-Expression $expr
}

Write-Host "Waiting for Round 6 eval + DINO to complete..."
for ($i = 0; $i -lt 60; $i++) {
    $evalExists = Invoke-RemoteEncoded "Test-Path '$EvalDone'"
    $dinoExists = Invoke-RemoteEncoded "Test-Path '$DinoDone'"
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] eval=$evalExists dino=$dinoExists"
    if (($evalExists -match "True") -and ($dinoExists -match "True")) {
        Write-Host "Results ready. Extracting..."
        break
    }
    Start-Sleep -Seconds 30
}

$extractCmd = "python I:\Github\Latent_Style\SchrodingerBridge\scripts\_extract_round6_results.py"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($extractCmd))
$expr = "ssh.exe -p 2222 -o LogLevel=ERROR $Remote `"powershell -EncodedCommand $encoded`""
Invoke-Expression $expr
