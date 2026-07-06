# Inspect wikiarts15_eval logs and metrics structure
$script = @'
$logsDir = "I:\Github\Latent_Style\SchrodingerBridge\logs"

Write-Output "=== wikiarts15_eval.log ==="
Get-Content "$logsDir\wikiarts15_eval.log" -Tail 50

Write-Output ""
Write-Output "=== wikiarts15_eval.log.out (tail) ==="
Get-Content "$logsDir\wikiarts15_eval.log.out" -Tail 50

Write-Output ""
Write-Output "=== wikiarts15_eval.log.err (tail) ==="
Get-Content "$logsDir\wikiarts15_eval.log.err" -Tail 30

Write-Output ""
Write-Output "=== metrics.csv header (first 2 lines) ==="
Get-Content "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts15_eval\metrics.csv" -TotalCount 2

Write-Output ""
Write-Output "=== metrics.csv line count ==="
(Get-Content "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts15_eval\metrics.csv" | Measure-Object -Line).Lines

Write-Output ""
Write-Output "=== check for any JSON summary files in wikiarts15_eval ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts15_eval" -Recurse -Filter "*.json" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Output ("{0} | {1}" -f $_.FullName, $_.Length)
}
'@

$bytes = [System.Text.Encoding]::Unicode.GetBytes($script)
$b64 = [Convert]::ToBase64String($bytes)

ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -EncodedCommand $b64"
