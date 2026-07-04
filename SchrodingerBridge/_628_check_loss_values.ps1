# Check the actual loss values in the training CSV log to verify auxiliary loss is REAL
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$expDir = "$root\exp\628_ablation\destructive\X1_velmag_w10"

Write-Host "=== X1_velmag_w10 experiment directory ==="
if (Test-Path $expDir) {
    Get-ChildItem $expDir -Recurse | Format-Table Name,Length,LastWriteTime
} else {
    Write-Host "Directory not found: $expDir"
}

# Check training CSV log
Write-Host "`n=== Training CSV log ==="
$csvLog = Get-ChildItem $expDir -Filter "*.csv" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if ($csvLog) {
    Write-Host "Found: $($csvLog.FullName)"
    # Read header and first few rows
    $content = Get-Content $csvLog.FullName -Head 5
    foreach ($line in $content) {
        Write-Host $line
    }
    # Search for velocity-related columns
    $header = $content[0]
    if ($header -match 'vel_mag|velocity|v_pred|v_target') {
        Write-Host "`nFOUND velocity-related columns in CSV!"
    } else {
        Write-Host "`nNo velocity columns in header"
        Write-Host "Header: $header"
    }
} else {
    Write-Host "No CSV log found"
}

# Check numeric debug
Write-Host "`n=== Numeric debug ==="
$numDebug = Get-ChildItem $expDir -Filter "numeric_debug*" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if ($numDebug) {
    Write-Host "Found: $($numDebug.FullName)"
    Get-Content $numDebug.FullName -Head 5
} else {
    Write-Host "No numeric debug file"
}

# Search for loss_velocity_magnitude in any log file
Write-Host "`n=== Searching for loss_velocity_magnitude in all logs ==="
$logDir = "$root\exp\628_ablation\destructive_logs"
$x1Log = "$logDir\X1_velmag_w10_verify.log"
if (Test-Path $x1Log) {
    $size = (Get-Item $x1Log).Length
    Write-Host "X1 log size: $size bytes"
    # Search for key patterns
    $patterns = @('vel_mag', 'velocity_magnitude', 'velocity_ratio', 'v_pred_norm', 'v_target_norm',
                  'loss_contrast', 'loss_output_var', 'loss_directional', 'loss_channel_var',
                  'loss_hf_energy', 'loss_pixel_color', 'loss_saturation', 'loss_attn_entropy')
    foreach ($p in $patterns) {
        $m = Select-String -Path $x1Log -Pattern $p -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($m) {
            Write-Host "  FOUND $p : $($m.Line.Substring(0, [Math]::Min(200, $m.Line.Length)))"
        }
    }
}

# Also check if the experiment is still running
Write-Host "`n=== Python processes ==="
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id,StartTime,CPU,@{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB,1)}}
