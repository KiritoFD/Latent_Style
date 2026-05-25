param(
    [string]$RepoRoot = "I:\Github\Latent_Style\SchrodingerBridge",
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$manifestRoot = Join-Path $RepoRoot "exp\_cleanup_manifests\vae_backend_cleanup_$timestamp"
New-Item -ItemType Directory -Force -Path $manifestRoot | Out-Null

$roots = @(
    "exp",
    "scale",
    "experiments",
    "orthogonal_phase_space_sweep_60",
    "review_additional_experiments",
    "high_tension_phase_space_sweep",
    "full_dimensional_orthogonal_sweep_20"
) | Where-Object { Test-Path $_ }

$protectedRegex = @(
    "vae_backend_256",
    "diffeomorphic_tangent_sweep",
    "frontier_decision_tree_8h",
    "orthogonal_budget36",
    "paper_main_750_bundle",
    "paper_metric_canonical",
    "paper_table_metric_supplement",
    "selected_epoch_curves",
    "video_compare"
) -join "|"

function Get-ExperimentRoot([System.IO.FileInfo]$File) {
    $dir = $File.Directory
    while ($null -ne $dir -and $dir.FullName.StartsWith($RepoRoot)) {
        if (Test-Path (Join-Path $dir.FullName "config.json")) {
            return $dir.FullName
        }
        $dir = $dir.Parent
    }
    return $File.Directory.FullName
}

function Get-EpochFromName([string]$Name) {
    if ($Name -match "epoch_(\d+)\.pt$") { return [int]$Matches[1] }
    return $null
}

function Read-EvalScores([string]$ExpRoot) {
    $rows = @()
    foreach ($summary in Get-ChildItem (Join-Path $ExpRoot "full_eval") -Recurse -Filter "summary.json" -File -ErrorAction SilentlyContinue) {
        $epoch = $null
        if ($summary.Directory.Name -match "epoch_(\d+)") { $epoch = [int]$Matches[1] }
        if ($null -eq $epoch) { continue }
        try {
            $payload = Get-Content $summary.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
            $overview = $payload.analysis.all_pairs_overview
            $style = [double]$overview.clip_style
            $lpips = [double]$overview.content_lpips
            $ec = $style * (1.0 - $lpips)
            $rows += [pscustomobject]@{ epoch=$epoch; clip_style=$style; content_lpips=$lpips; ec=$ec; summary=$summary.FullName }
        } catch {
        }
    }
    return $rows
}

$candidates = New-Object System.Collections.Generic.List[object]
$preserved = New-Object System.Collections.Generic.List[object]

foreach ($root in $roots) {
    foreach ($dir in Get-ChildItem $root -Recurse -Directory -Filter "images" -ErrorAction SilentlyContinue) {
        if ($dir.FullName -notmatch "\\full_eval\\") { continue }
        if ($dir.FullName -match $protectedRegex) {
            $preserved.Add([pscustomobject]@{ path=$dir.FullName; type="dir"; reason="protected_experiment_images" })
            continue
        }
        $epochDir = $dir.Parent.FullName
        $hasLedger = (Test-Path (Join-Path $epochDir "summary.json")) -or (Test-Path (Join-Path $epochDir "metrics.csv"))
        if (-not $hasLedger) {
            $preserved.Add([pscustomobject]@{ path=$dir.FullName; type="dir"; reason="full_eval_images_without_summary_or_metrics" })
            continue
        }
        $bytes = (Get-ChildItem $dir.FullName -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        $candidates.Add([pscustomobject]@{ path=$dir.FullName; type="dir"; bytes=[int64]$bytes; reason="full_eval_images_with_summary_or_metrics" })
    }
}

$ckptsByExp = @{}
foreach ($root in $roots) {
    foreach ($ckpt in Get-ChildItem $root -Recurse -Filter "epoch_*.pt" -File -ErrorAction SilentlyContinue) {
        $expRoot = Get-ExperimentRoot $ckpt
        if (-not $ckptsByExp.ContainsKey($expRoot)) { $ckptsByExp[$expRoot] = @() }
        $ckptsByExp[$expRoot] = @($ckptsByExp[$expRoot]) + @($ckpt)
    }
}

foreach ($expRoot in $ckptsByExp.Keys) {
    $ckpts = @($ckptsByExp[$expRoot])
    $keepEpochs = New-Object System.Collections.Generic.HashSet[int]
    $epochs = @($ckpts | ForEach-Object { Get-EpochFromName $_.Name } | Where-Object { $null -ne $_ })
    if ($epochs.Count -gt 0) { [void]$keepEpochs.Add(($epochs | Measure-Object -Maximum).Maximum) }
    foreach ($row in Read-EvalScores $expRoot) {
        # Filled below by ranking after rows are collected.
    }
    $scoreRows = @(Read-EvalScores $expRoot)
    if ($scoreRows.Count -gt 0) {
        [void]$keepEpochs.Add((@($scoreRows | Sort-Object ec -Descending | Select-Object -First 1)[0]).epoch)
        [void]$keepEpochs.Add((@($scoreRows | Sort-Object clip_style -Descending | Select-Object -First 1)[0]).epoch)
        [void]$keepEpochs.Add((@($scoreRows | Sort-Object content_lpips | Select-Object -First 1)[0]).epoch)
    }
    $isProtected = $expRoot -match $protectedRegex
    foreach ($ckpt in $ckpts) {
        $epoch = Get-EpochFromName $ckpt.Name
        if ($isProtected) {
            $preserved.Add([pscustomobject]@{ path=$ckpt.FullName; type="file"; reason="protected_experiment" })
            continue
        }
        if ($null -ne $epoch -and $keepEpochs.Contains([int]$epoch)) {
            $preserved.Add([pscustomobject]@{ path=$ckpt.FullName; type="file"; reason="latest_or_best_metric_epoch" })
            continue
        }
        $hasContext = (Test-Path (Join-Path $expRoot "config.json")) -and (
            (Test-Path (Join-Path $expRoot "logs")) -or
            (Test-Path (Join-Path $expRoot "full_eval")) -or
            (Get-ChildItem $expRoot -Filter "*.csv" -File -ErrorAction SilentlyContinue | Select-Object -First 1)
        )
        if ($hasContext) {
            $candidates.Add([pscustomobject]@{ path=$ckpt.FullName; type="file"; bytes=[int64]$ckpt.Length; reason="non_representative_ckpt_with_config_and_results" })
        } else {
            $preserved.Add([pscustomobject]@{ path=$ckpt.FullName; type="file"; reason="ckpt_without_sufficient_ledger_context" })
        }
    }
}

$candidateCsv = Join-Path $manifestRoot "delete_candidates.csv"
$preservedCsv = Join-Path $manifestRoot "preserved_files.csv"
$candidates | Sort-Object bytes -Descending | Export-Csv $candidateCsv -NoTypeInformation -Encoding UTF8
$preserved | Export-Csv $preservedCsv -NoTypeInformation -Encoding UTF8

$totalBytes = [int64](($candidates | Measure-Object bytes -Sum).Sum)
$summary = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    apply = [bool]$Apply
    repo_root = $RepoRoot
    roots = $roots
    protected_regex = $protectedRegex
    candidate_count = $candidates.Count
    preserved_count = $preserved.Count
    candidate_gb = [math]::Round($totalBytes / 1GB, 3)
    candidate_csv = $candidateCsv
    preserved_csv = $preservedCsv
}
$summary | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $manifestRoot "summary.json") -Encoding UTF8

if ($Apply) {
    foreach ($item in $candidates) {
        if ($item.type -eq "dir") {
            Remove-Item -LiteralPath $item.path -Recurse -Force -ErrorAction Continue
        } elseif ($item.type -eq "file") {
            Remove-Item -LiteralPath $item.path -Force -ErrorAction Continue
        }
    }
}

Write-Output ($summary | ConvertTo-Json -Depth 6)
