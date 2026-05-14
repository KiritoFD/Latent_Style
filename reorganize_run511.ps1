$rw = "G:\GitHub\Latent_Style\Related_Works\run_511"

New-Item -ItemType Directory -Path "$rw\eval" -Force | Out-Null
New-Item -ItemType Directory -Path "$rw\launchers" -Force | Out-Null
New-Item -ItemType Directory -Path "$rw\summaries" -Force | Out-Null
New-Item -ItemType Directory -Path "$rw\diagnostics" -Force | Out-Null

# --- eval scripts ---
$eval_scripts = @("eval_750.py","eval_adain.py","eval_antihf_750.py","eval_artifact_pack_750.py","eval_guard_750.py","eval_hf_patch_kid_750.py","eval_plain_kid_750.py")
foreach ($f in $eval_scripts) {
    $src = "$rw\$f"
    if (Test-Path $src) {
        Write-Host "MOVE eval: $f"
        Move-Item $src "$rw\eval\$f" -Force
    }
}

# --- launchers (run_*.py, run_*.bat, smoke_*.bat) ---
Get-ChildItem $rw -Filter "run_*" -File | ForEach-Object {
    Write-Host "MOVE launcher: $($_.Name)"
    Move-Item $_.FullName "$rw\launchers\$($_.Name)" -Force
}
Get-ChildItem $rw -Filter "smoke_*" -File | ForEach-Object {
    Write-Host "MOVE launcher: $($_.Name)"
    Move-Item $_.FullName "$rw\launchers\$($_.Name)" -Force
}

# --- summaries ---
$summaries = @("prepare_complete_750.py","make_protocol750_report.py","build_timing_comparison.py","fill_missing_timing.py","summarize_artifact_pack_750.py","summarize_complete_750.py","summarize_outputs.py","summarize_stroke_grain_750.py","summarize_timing.py")
foreach ($f in $summaries) {
    $src = "$rw\$f"
    if (Test-Path $src) {
        Write-Host "MOVE summary: $f"
        Move-Item $src "$rw\summaries\$f" -Force
    }
}

# --- diagnostics ---
$diag = @("diagnose_samst_outputs.py","diagnostic_samst_contact.jpg","diagnostic_samst_random25.jpg","diagnostic_samst_stats.md")
foreach ($f in $diag) {
    $src = "$rw\$f"
    if (Test-Path $src) {
        Write-Host "MOVE diag: $f"
        Move-Item $src "$rw\diagnostics\$f" -Force
    }
}

# --- move remaining loose files to docs/ ---
New-Item -ItemType Directory -Path "$rw\docs" -Force | Out-Null
$docs = @("README.md","STATUS_AND_PLAN.md","outputs_inventory.md","outputs_inventory.csv","timing_summary.md","timing_summary.csv","protocol750_eval_report.md","protocol750_eval_report.csv","timing_metrics_combined.json","timing_comparison_ours_vs_samst.json")
foreach ($f in $docs) {
    $src = "$rw\$f"
    if (Test-Path $src) {
        Write-Host "MOVE doc: $f"
        Move-Item $src "$rw\docs\$f" -Force
    }
}

Write-Host "`nDONE"
