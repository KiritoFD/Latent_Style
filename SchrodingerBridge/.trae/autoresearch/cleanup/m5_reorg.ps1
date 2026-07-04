$ErrorActionPreference = 'Stop'
Set-Location 'g:\GitHub\Latent_Style\SchrodingerBridge'

$logFile = '.trae\autoresearch\cleanup\logs\m5_reorg.log'
Set-Content -Path $logFile -Value "=== M5 Reorganization Log - $(Get-Date) ==="

function Move-DirLogged {
    param([string]$Src, [string]$DestParent)
    if (Test-Path $Src) {
        if (-not (Test-Path $DestParent)) { New-Item -ItemType Directory -Path $DestParent -Force | Out-Null }
        $name = Split-Path $Src -Leaf
        $dest = Join-Path $DestParent $name
        if (Test-Path $dest) {
            Write-Host "SKIP $Src -> $dest (already exists)"
            Add-Content -Path $logFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP $Src -> $dest (already exists)"
            return
        }
        try {
            Move-Item -Path $Src -Destination $DestParent -ErrorAction Stop
            $entry = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] MOVED $Src -> $dest"
            Write-Host $entry
            Add-Content -Path $logFile -Value $entry
        } catch {
            $entry = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAILED $Src -> $dest | $_"
            Write-Host $entry
            Add-Content -Path $logFile -Value $entry
        }
    } else {
        Add-Content -Path $logFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP $Src (not found)"
    }
}

# === Create target directories ===
$targets = @(
    'exp\exp_baselines',
    'exp\exp_ours\early',
    'exp\exp_ours\phase4',
    'exp\exp_ours\local_t',
    'exp\exp_shared'
)
foreach ($t in $targets) {
    if (-not (Test-Path $t)) { New-Item -ItemType Directory -Path $t -Force | Out-Null }
}

# === 1. Baselines ===
Write-Host "`n=== Moving baselines ==="
$baselines = @('baseline_reeval', 'baseline_images', 'baseline_v2')
foreach ($d in $baselines) {
    Move-DirLogged "exp\$d" 'exp\exp_baselines'
}

# === 2. Shared resources ===
Write-Host "`n=== Moving shared resources ==="
$shared = @('adain_checkpoints', 'eval_cache', 'clean_base')
foreach ($d in $shared) {
    Move-DirLogged "exp\$d" 'exp\exp_shared'
}
# Eval-only result dirs (no config, no src, just eval outputs)
$evalOnly = @('630_local_t3_eval_ll005', '630_local_t4_eval', '630_local_t12_eval', '630_planA_zero_step_wct')
foreach ($d in $evalOnly) {
    Move-DirLogged "exp\$d" 'exp\exp_shared'
}

# === 3. Early experiments (task1-6, clean_base_v2, 628, phase3_task2) ===
Write-Host "`n=== Moving early experiments ==="
$early = @(
    'task1_endpoint_film_baseline',
    'task1_endpoint_film_no_norm',
    'task3_baseline_1ep',
    'task3_combo_a_1ep',
    'task3_combo_b_3ep',
    'task4_iter',
    'task4_style_strength_baseline_2ep',
    'task4_style_strength_w05_2ep',
    'task4_style_strength_w10_2ep',
    'phase3_task2_p3d_contrastive_w01_margin01',
    'phase3_task2_p3e_contrastive_w05_margin005',
    'clean_base_v2_local',
    'clean_base_v2_relu2',
    '628_ablation'
)
foreach ($d in $early) {
    Move-DirLogged "exp\$d" 'exp\exp_ours\early'
}

# === 4. 630_phase4* series ===
Write-Host "`n=== Moving 630_phase4* series ==="
Get-ChildItem -Path 'exp' -Directory -Filter '630_phase4*' | ForEach-Object {
    Move-DirLogged $_.FullName 'exp\exp_ours\phase4'
}

# === 5. 630_phase1d/2b/2c/3 series (phase1-3, before phase4) ===
Write-Host "`n=== Moving 630_phase1d/2b/2c/3 series ==="
Get-ChildItem -Path 'exp' -Directory -Filter '630_phase1d*' | ForEach-Object {
    Move-DirLogged $_.FullName 'exp\exp_ours\phase4'
}
Get-ChildItem -Path 'exp' -Directory -Filter '630_phase2*' | ForEach-Object {
    Move-DirLogged $_.FullName 'exp\exp_ours\phase4'
}
Get-ChildItem -Path 'exp' -Directory -Filter '630_phase3*' | ForEach-Object {
    Move-DirLogged $_.FullName 'exp\exp_ours\phase4'
}

# === 6. 630_local_* T-series + R-series ===
Write-Host "`n=== Moving 630_local_* T/R series ==="
Get-ChildItem -Path 'exp' -Directory -Filter '630_local_*' | ForEach-Object {
    Move-DirLogged $_.FullName 'exp\exp_ours\local_t'
}

# === Final: list remaining at exp/ root ===
Write-Host "`n=== Remaining at exp/ root ==="
$remaining = Get-ChildItem -Path 'exp' -Directory | Where-Object { $_.Name -notin @('exp_baselines', 'exp_ours', 'exp_shared') }
if ($remaining) {
    $remaining | ForEach-Object { Write-Host "  $($_.Name)"; Add-Content -Path $logFile -Value "REMAINING: $($_.Name)" }
} else {
    Write-Host "  (none - all moved)"
}

Write-Host "`n=== Reorganization complete ==="
Write-Host "Structure:"
Write-Host "  exp/exp_baselines/  - baseline evaluation results"
Write-Host "  exp/exp_ours/early/ - early task1-6 + clean_base + 628"
Write-Host "  exp/exp_ours/phase4/ - 630_phase1d/2/3/4 series (66 dirs)"
Write-Host "  exp/exp_ours/local_t/ - 630_local T/R series (24 dirs)"
Write-Host "  exp/exp_shared/    - shared eval caches + adain checkpoints"
