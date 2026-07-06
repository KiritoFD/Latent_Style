# Extract aggregate metrics from summary.json on remote
$script = @'
$sumPath = "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts15_eval\summary.json"
$csvPath = "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts15_eval\metrics.csv"

# Approach 1: Try to read summary.json top-level keys
Write-Output "=== summary.json top-level structure ==="
$json = Get-Content $sumPath -Raw | ConvertFrom-Json
Write-Output ("Top-level keys: " + ($json.PSObject.Properties.Name -join ", "))

# Try common aggregate fields
foreach ($key in @("transfer", "allpairs", "all_pairs", "aggregate", "metrics", "summary", "overall")) {
    if ($json.PSObject.Properties.Name -contains $key) {
        Write-Output ("--- {0} ---" -f $key)
        $json.$key | ConvertTo-Json -Depth 5
    }
}

# Approach 2: Compute from CSV directly (more reliable)
Write-Output ""
Write-Output "=== Aggregate metrics from CSV ==="
Import-Csv $csvPath | ForEach-Object {
    [PSCustomObject]@{
        src_style = $_.src_style
        tgt_style = $_.tgt_style
        lpips = [double]$_.content_lpips
        clip_s = [double]$_.clip_style
        is_identity = ($_.src_style -eq $_.tgt_style)
    }
} | Group-Object -Property is_identity | ForEach-Object {
    $name = if ($_.Name -eq "True") { "identity" } else { "transfer" }
    $lpips_avg = ($_.Group | Measure-Object -Property lpips -Average).Average
    $clip_avg = ($_.Group | Measure-Object -Property clip_s -Average).Average
    $count = $_.Count
    Write-Output ("{0}: count={1}, CLIP-S={2:.4f}, LPIPS={3:.4f}" -f $name, $count, $clip_avg, $lpips_avg)
}

# All-pairs (all rows)
Write-Output ""
Write-Output "=== ALL-PAIRS (all rows) ==="
$all = Import-Csv $csvPath | ForEach-Object {
    [PSCustomObject]@{
        lpips = [double]$_.content_lpips
        clip_s = [double]$_.clip_style
    }
}
$lpips_all = ($all | Measure-Object -Property lpips -Average).Average
$clip_all = ($all | Measure-Object -Property clip_s -Average).Average
Write-Output ("allpairs: count={0}, CLIP-S={1:.4f}, LPIPS={2:.4f}" -f $all.Count, $clip_all, $lpips_all)
'@

$bytes = [System.Text.Encoding]::Unicode.GetBytes($script)
$b64 = [Convert]::ToBase64String($bytes)

ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -EncodedCommand $b64"
