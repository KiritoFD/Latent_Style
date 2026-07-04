# Check N1 training CSV log to see where it failed
$base = "I:\Github\Latent_Style\SchrodingerBridge"
$n1Dir = "$base\exp\p4_fusion_breakout\n1_lvl2_gate03_whh25"
$logsDir = "$n1Dir\logs"

Write-Host "=== N1 Training CSV Logs ==="
if (Test-Path $logsDir) {
    $csvFiles = Get-ChildItem $logsDir -Filter "*.csv"
    foreach ($f in $csvFiles) {
        Write-Host ""
        Write-Host "--- $($f.Name) (size=$($f.Length) bytes) ---"
        $content = Get-Content $f.FullName -ErrorAction SilentlyContinue
        if ($content) {
            Write-Host "Total lines: $($content.Count)"
            Write-Host "--- First 3 lines (header + data) ---"
            $content | Select-Object -First 3 | ForEach-Object {
                # Truncate long lines
                if ($_.Length -gt 300) {
                    Write-Host $_.Substring(0, 300) + "..."
                } else {
                    Write-Host $_
                }
            }
            if ($content.Count -gt 3) {
                Write-Host "--- Last 3 lines ---"
                $content | Select-Object -Last 3 | ForEach-Object {
                    if ($_.Length -gt 300) {
                        Write-Host $_.Substring(0, 300) + "..."
                    } else {
                        Write-Host $_
                    }
                }
            }
        } else {
            Write-Host "[EMPTY] File has no content"
        }
    }
} else {
    Write-Host "[WARN] logs dir not found"
}

Write-Host ""
Write-Host "=== N1 config.json verification ==="
$configFile = "$n1Dir\config.json"
if (Test-Path $configFile) {
    $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
    Write-Host "spectral_ode_levels = $($cfg.model.spectral_ode_levels)"
    Write-Host "style_cross_attn_gate_init = $($cfg.model.style_cross_attn_gate_init)"
    Write-Host "spectral_w_hh = $($cfg.bridge.spectral_w_hh)"
    Write-Host "num_epochs = $($cfg.training.num_epochs)"
    Write-Host "full_eval_each_epoch = $($cfg.training.full_eval_each_epoch)"
}

Write-Host ""
Write-Host "=== Check for any error/traceback in log files ==="
$allLogs = Get-ChildItem $n1Dir -Recurse -Filter "*.log" -ErrorAction SilentlyContinue
if ($allLogs) {
    foreach ($f in $allLogs) {
        Write-Host "--- $($f.Name) (size=$($f.Length)) ---"
        if ($f.Length -gt 0) {
            Get-Content $f.FullName -Tail 30
        }
    }
} else {
    Write-Host "No .log files found"
}
