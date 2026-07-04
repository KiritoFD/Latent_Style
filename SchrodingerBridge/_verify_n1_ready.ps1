# Verify N1 multi-level DWT training readiness
$base = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Host "=== N1 Config Check ==="
$n1Config = "$base\configs\p4_n1_lvl2.json"
if (Test-Path $n1Config) {
    Write-Host "[OK] N1_CONFIG_EXISTS: $n1Config"
    $cfg = Get-Content $n1Config -Raw | ConvertFrom-Json
    Write-Host "  spectral_ode_levels = $($cfg.model.spectral_ode_levels)"
    Write-Host "  style_cross_attn_gate_init = $($cfg.model.style_cross_attn_gate_init)"
    Write-Host "  spectral_w_hh = $($cfg.bridge.spectral_w_hh)"
    Write-Host "  num_epochs = $($cfg.training.num_epochs)"
    Write-Host "  full_eval_each_epoch = $($cfg.training.full_eval_each_epoch)"
    Write-Host "  save_dir = $($cfg.checkpoint.save_dir)"
} else {
    Write-Host "[FAIL] N1_CONFIG_MISSING"
}

Write-Host ""
Write-Host "=== Training Script Check ==="
$trainScript = "$base\_run_train_schtasks.ps1"
if (Test-Path $trainScript) {
    Write-Host "[OK] SCHTASKS_SCRIPT_EXISTS"
} else {
    Write-Host "[FAIL] SCHTASKS_SCRIPT_MISSING"
}

Write-Host ""
Write-Host "=== spectral_bridge620.py Key Code Check ==="
$sbFile = "$base\src\spectral_bridge620.py"
if (Test-Path $sbFile) {
    $lines = Get-Content $sbFile
    $spectralLevelsLines = @()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match 'spectral_levels') {
            $spectralLevelsLines += "$($i+1): $($lines[$i].Trim())"
        }
    }
    Write-Host "spectral_levels occurrences ($($spectralLevelsLines.Count) lines):"
    $spectralLevelsLines | ForEach-Object { Write-Host "  $_" }

    $multiLevelLines = @()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match 'dwt2_multi_level|idwt2_multi_level') {
            $multiLevelLines += "$($i+1): $($lines[$i].Trim())"
        }
    }
    Write-Host ""
    Write-Host "multi_level DWT calls ($($multiLevelLines.Count) lines):"
    $multiLevelLines | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "[FAIL] spectral_bridge620.py MISSING"
}

Write-Host ""
Write-Host "=== Running Python Processes ==="
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcs) {
    Write-Host "[WARN] Found $($pythonProcs.Count) Python process(es):"
    $pythonProcs | ForEach-Object { Write-Host "  PID=$($_.Id) StartTime=$($_.StartTime)" }
} else {
    Write-Host "[OK] No running Python process, ready to start N1 training"
}

Write-Host ""
Write-Host "=== N11+N16 Training Results Check ==="
$n11Dir = "$base\exp\p4_fusion_breakout\n11_n16_gate03_whh25"
if (Test-Path $n11Dir) {
    $ckptDir = "$n11Dir\checkpoints"
    if (Test-Path $ckptDir) {
        $ckpts = Get-ChildItem $ckptDir -Filter "*.pt"
        Write-Host "[OK] N11+N16 checkpoints ($($ckpts.Count) files):"
        $ckpts | ForEach-Object { Write-Host "  $($_.Name)" }
    } else {
        Write-Host "[WARN] N11+N16 checkpoints dir missing"
    }
} else {
    Write-Host "[WARN] N11+N16 experiment dir missing"
}
