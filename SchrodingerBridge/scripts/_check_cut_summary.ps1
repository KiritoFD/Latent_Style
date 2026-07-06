# Check CUT summary.json structure
$ErrorActionPreference = "Continue"
$sum = "I:\Github\Latent_Style\final_works\CUT\summary.json"
if (Test-Path $sum) {
    $j = Get-Content $sum -Raw | ConvertFrom-Json
    Write-Host "=== Top-level keys ==="
    $j.PSObject.Properties | ForEach-Object { Write-Host "  $($_.Name): $($_.Value.GetType().Name)" }

    Write-Host ""
    Write-Host "=== matrix_breakdown sample ==="
    if ($j.matrix_breakdown) {
        $j.matrix_breakdown | Get-Member -MemberType NoteProperty | Select-Object -First 5 | ForEach-Object {
            $key = $_.Name
            Write-Host "--- $key ---"
            $j.matrix_breakdown.$key | ConvertTo-Json -Depth 3
        }
    }

    Write-Host ""
    Write-Host "=== metrics_note ==="
    if ($j.metrics_note) { Write-Host $j.metrics_note }

    Write-Host ""
    Write-Host "=== analysis (first 500 chars) ==="
    if ($j.analysis) {
        $a = $j.analysis | ConvertTo-Json -Depth 5
        if ($a.Length -gt 500) { $a = $a.Substring(0, 500) + "..." }
        Write-Host $a
    }
}
