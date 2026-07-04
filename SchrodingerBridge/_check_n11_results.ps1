# Check N11+N16 experiment directory structure and start N1 training
$base = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Host "=== N11+N16 Experiment Dir Structure ==="
$n11Dir = "$base\exp\p4_fusion_breakout\n11_n16_gate03_whh25"
if (Test-Path $n11Dir) {
    Write-Host "[OK] N11+N16 dir exists: $n11Dir"
    Write-Host "Contents:"
    Get-ChildItem $n11Dir -Recurse -Depth 2 | ForEach-Object {
        $rel = $_.FullName.Replace($n11Dir, ".")
        Write-Host "  $rel"
    }
} else {
    Write-Host "[FAIL] N11+N16 dir missing: $n11Dir"
    Write-Host "Checking parent dir exp/p4_fusion_breakout/:"
    $parent = "$base\exp\p4_fusion_breakout"
    if (Test-Path $parent) {
        Get-ChildItem $parent -Directory | ForEach-Object { Write-Host "  $($_.Name)" }
    } else {
        Write-Host "  parent dir also missing"
    }
}

Write-Host ""
Write-Host "=== Check N11+N16 eval results (summary.json) ==="
$n11EvalDirs = @(
    "$base\exp\p4_fusion_breakout\n11_n16_gate03_whh25",
    "$base\exp\p4_fusion_breakout"
)
foreach ($d in $n11EvalDirs) {
    if (Test-Path $d) {
        $summaries = Get-ChildItem $d -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue
        if ($summaries) {
            Write-Host "Found $($summaries.Count) summary.json in $d"
            $summaries | Select-Object -First 10 | ForEach-Object {
                Write-Host "  $($_.FullName.Replace($base, ''))"
            }
        }
    }
}
