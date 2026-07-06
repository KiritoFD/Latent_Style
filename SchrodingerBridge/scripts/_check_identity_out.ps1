# Check identity gen output and wait for adain progress
$script = @'
$repo = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Output "=== identity gen.out ==="
Get-Content "$repo\logs\baseline_wikiarts15.log.identity.gen.out"

Write-Output ""
Write-Output "=== identity gen.err (if any) ==="
$errFile = "$repo\logs\baseline_wikiarts15.log.identity.gen.err"
if ((Get-Item $errFile).Length -gt 0) {
    Get-Content $errFile
} else {
    Write-Output "(empty)"
}

Write-Output ""
Write-Output "=== adain gen.out so far ==="
Get-Content "$repo\logs\baseline_wikiarts15.log.adain.gen.out"

Write-Output ""
Write-Output "=== adain image count ==="
$adainDir = "$repo\exp\baseline_wikiarts15\adain\images"
if (Test-Path $adainDir) {
    $cnt = (Get-ChildItem $adainDir -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Output "adain images: $cnt / 6750"
}
'@

$bytes = [System.Text.Encoding]::Unicode.GetBytes($script)
$b64 = [Convert]::ToBase64String($bytes)

ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -EncodedCommand $b64"
