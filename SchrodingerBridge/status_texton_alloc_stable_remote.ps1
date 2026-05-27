$ErrorActionPreference = "Continue"
Set-Location 'I:\Github\Latent_Style\SchrodingerBridge'

$task = 'LANCET_transport_texton_alloc_stable'
$root = 'exp\vae_backend\ema_transport_texton_alloc_stable'

Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue |
    Select-Object TaskName, State |
    Format-List |
    Out-String -Width 200

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

if (Test-Path "$root\task.log") {
    "--- task.log ---"
    Get-Content "$root\task.log" -Tail 30
}

if (Test-Path "$root\vae_backend_256_results.csv") {
    "--- csv ---"
    Get-Content "$root\vae_backend_256_results.csv" -Tail 20
}

"--- latest cross_by_target_style ---"
$summaries = Get-ChildItem $root -Recurse -Filter summary.json -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 6
foreach ($summary in $summaries) {
    "summary: $($summary.FullName)"
    $payload = Get-Content $summary.FullName -Raw | ConvertFrom-Json
    $cross = $payload.analysis.cross_by_target_style
    if ($null -ne $cross) {
        $cross.PSObject.Properties |
            Sort-Object Name |
            ForEach-Object {
                $v = $_.Value
                "{0} cross style={1:N5} lpips={2:N5} images={3}" -f $_.Name, [double]$v.clip_style, [double]$v.content_lpips, [int]$v.count
            }
    }
}
