$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = 'I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge'
$expRoot = Join-Path $root 'exp'

function SafeVal($Value) {
    if ($null -eq $Value) { return '' }
    return [string]$Value
}

function Round3($Value) {
    if ($null -eq $Value -or $Value -eq '') { return '' }
    try { return ('{0:N3}' -f [double]$Value) } catch { return [string]$Value }
}

function OneLine($Text) {
    if ($null -eq $Text) { return '' }
    return (($Text -replace "`r", ' ' -replace "`n", ' ' -replace '"', "'") -replace '\s+', ' ').Trim()
}

$rows = New-Object System.Collections.Generic.List[object]
$dirs = @(Get-ChildItem -LiteralPath $expRoot -Directory | Sort-Object Name)

foreach ($dir in $dirs) {
    $files = @(Get-ChildItem -LiteralPath $dir.FullName -Recurse -File -ErrorAction SilentlyContinue)
    $sizeMb = (($files | Measure-Object Length -Sum).Sum / 1MB)
    $weights = @($files | Where-Object { $_.Extension -in '.pt', '.ckpt', '.pth' })
    $summaries = @($files | Where-Object { $_.Name -eq 'summary.json' } | Sort-Object FullName)

    $cfgPath = Join-Path $dir.FullName 'config.json'
    $cfg = $null
    if (Test-Path $cfgPath) {
        try {
            $cfg = Get-Content -LiteralPath $cfgPath -Encoding UTF8 -Raw | ConvertFrom-Json
        } catch {
            $cfg = $null
        }
    }

    $summaryBits = New-Object System.Collections.Generic.List[string]
    foreach ($summary in $summaries) {
        try {
            $json = Get-Content -LiteralPath $summary.FullName -Encoding UTF8 -Raw | ConvertFrom-Json
            $epoch = Split-Path (Split-Path $summary.FullName -Parent) -Leaf
            $clip = $json.analysis.all_pairs_overview.clip_style
            $lpips = $json.analysis.all_pairs_overview.content_lpips
            $wall = $json.timings_sec.wall_total
            $summaryBits.Add(('{0}:clip={1};lpips={2};wall={3}' -f $epoch, (Round3 $clip), (Round3 $lpips), (Round3 $wall)))
        } catch {
            $summaryBits.Add((Split-Path (Split-Path $summary.FullName -Parent) -Leaf) + ':parse_error')
        }
    }

    $trainCsvs = @($files | Where-Object { $_.Name -like 'training_*.csv' } | Sort-Object LastWriteTime)
    $latestTrainTail = ''
    if ($trainCsvs.Count -gt 0) {
        $lastCsv = $trainCsvs[-1]
        $tail = @(Get-Content -LiteralPath $lastCsv.FullName -Encoding UTF8 -Tail 3 | Where-Object { $_ -and ($_ -notmatch '^epoch,') })
        if ($tail.Count -gt 0) {
            $latestTrainTail = OneLine $tail[-1]
        }
    }

    $remoteLog = Join-Path $dir.FullName 'remote_train.log'
    $logTail = ''
    $params = ''
    if (Test-Path $remoteLog) {
        try {
            $tailLines = @(Get-Content -LiteralPath $remoteLog -Encoding UTF8 -Tail 12 | Where-Object { $_.Trim().Length -gt 0 })
            $logTail = OneLine (($tailLines | Select-Object -Last 4) -join ' | ')
            $paramLine = Select-String -LiteralPath $remoteLog -Pattern 'Model params:' -CaseSensitive:$false | Select-Object -Last 1
            if ($paramLine) {
                $params = (($paramLine.Line -replace '^.*Model params:\s*', '') -replace '[^0-9]', '')
            }
        } catch {
            $logTail = 'remote_train_log_parse_error'
        }
    }

    $topNames = (Get-ChildItem -LiteralPath $dir.FullName -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -First 25 -ExpandProperty Name) -join ';'

    $rows.Add([pscustomobject]@{
        remote_root = $root
        exp_dir = $dir.Name
        last_write = $dir.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
        total_mb = ('{0:N3}' -f $sizeMb)
        weight_count = $weights.Count
        weight_mb = ('{0:N3}' -f (($weights | Measure-Object Length -Sum).Sum / 1MB))
        weight_names = (($weights | Sort-Object Name | Select-Object -First 40 | ForEach-Object { $_.Name }) -join ';')
        config_exists = [bool](Test-Path $cfgPath)
        data_root = OneLine $cfg.data.data_root
        batch_size = SafeVal $cfg.training.batch_size
        num_epochs = SafeVal $cfg.training.num_epochs
        style_tokenizer = SafeVal $cfg.model.style_tokenizer
        style_spatial_mode = SafeVal $cfg.model.style_spatial_mode
        objective_mode = SafeVal $cfg.bridge.objective_mode
        loss_type = SafeVal $cfg.bridge.loss_type
        w_flow = SafeVal $cfg.bridge.w_flow
        terminal_swd_weight = SafeVal $cfg.bridge.terminal_swd_weight
        notes = OneLine $cfg.experiment.notes
        params_raw = $params
        summary_count = $summaries.Count
        summary_metrics = ($summaryBits -join ' | ')
        training_csv_count = $trainCsvs.Count
        latest_training_csv = if ($trainCsvs.Count -gt 0) { $trainCsvs[-1].FullName.Substring($root.Length + 1) } else { '' }
        latest_training_tail = $latestTrainTail
        remote_train_log_exists = [bool](Test-Path $remoteLog)
        remote_train_log_tail = $logTail
        top_level = $topNames
    })
}

$rows | ConvertTo-Csv -NoTypeInformation
