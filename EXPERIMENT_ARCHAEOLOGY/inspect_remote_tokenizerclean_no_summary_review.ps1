param(
    [Parameter(Mandatory = $true)]
    [string]$PolicyCsv,

    [string]$Root = 'I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$expRoot = Join-Path $Root 'exp'
$policy = Import-Csv -LiteralPath $PolicyCsv |
    Where-Object { $_.action -eq 'review_ckpt_candidate_no_summary' }

function OneLine($Text) {
    if ($null -eq $Text) { return '' }
    return (($Text -replace "`r", ' ' -replace "`n", ' ' -replace '"', "'") -replace '\s+', ' ').Trim()
}

function SafeVal($Value) {
    if ($null -eq $Value) { return '' }
    return [string]$Value
}

$rows = New-Object System.Collections.Generic.List[object]

foreach ($row in $policy) {
    $dirPath = Join-Path $expRoot $row.exp_dir
    if (!(Test-Path $dirPath)) {
        $rows.Add([pscustomobject]@{
            exp_dir = $row.exp_dir
            exists = $false
            review_class = 'missing_after_prior_cleanup'
            recommended_action = 'no_action'
            reason = 'Directory no longer exists.'
        })
        continue
    }

    $files = @(Get-ChildItem -LiteralPath $dirPath -Recurse -File -ErrorAction SilentlyContinue)
    $weights = @($files | Where-Object { $_.Extension -in '.pt', '.ckpt', '.pth' } | Sort-Object FullName)
    $summaries = @($files | Where-Object { $_.Name -eq 'summary.json' })
    $trainCsvs = @($files | Where-Object { $_.Name -like 'training_*.csv' } | Sort-Object LastWriteTime)
    $remoteLog = Join-Path $dirPath 'remote_train.log'
    $cfgPath = Join-Path $dirPath 'config.json'

    $cfg = $null
    if (Test-Path $cfgPath) {
        try {
            $cfg = Get-Content -LiteralPath $cfgPath -Encoding UTF8 -Raw | ConvertFrom-Json
        } catch {
            $cfg = $null
        }
    }

    $latestTrainingTail = ''
    if ($trainCsvs.Count -gt 0) {
        $tail = @(Get-Content -LiteralPath $trainCsvs[-1].FullName -Encoding UTF8 -Tail 4 |
            Where-Object { $_ -and ($_ -notmatch '^epoch,') })
        if ($tail.Count -gt 0) {
            $latestTrainingTail = OneLine $tail[-1]
        }
    }

    $remoteHead = ''
    $remoteTail = ''
    $hasTraceback = $false
    $hasComplete = $false
    $hasModelParams = ''
    if (Test-Path $remoteLog) {
        $all = @(Get-Content -LiteralPath $remoteLog -Encoding UTF8 -ErrorAction SilentlyContinue)
        $remoteHead = OneLine (($all | Select-Object -First 8) -join ' | ')
        $remoteTail = OneLine (($all | Select-Object -Last 12) -join ' | ')
        $hasTraceback = [bool]($all | Select-String -Pattern 'Traceback|CUDA out of memory|FileNotFoundError|No such file|could not open' -CaseSensitive:$false)
        $hasComplete = [bool]($all | Select-String -Pattern 'Training completed|completed|Finished|Saved checkpoint|Epoch .*complete' -CaseSensitive:$false)
        $paramLine = $all | Select-String -Pattern 'Model params:' -CaseSensitive:$false | Select-Object -Last 1
        if ($paramLine) {
            $hasModelParams = (($paramLine.Line -replace '^.*Model params:\s*', '') -replace '[^0-9]', '')
        }
    }

    $name = $row.exp_dir
    $reviewClass = 'manual_review_required'
    $recommended = 'keep_until_owner_review'
    $reason = 'No summary.json found, so the checkpoint may be the only payload.'

    if ($summaries.Count -gt 0) {
        $reviewClass = 'has_summary_after_all'
        $recommended = 'candidate_checkpoint_delete'
        $reason = 'A summary.json exists in current rescan.'
    } elseif ($name -like '*smoke*' -or $name -like '*calib*' -or $name -like '*120b*' -or $name -like '*20b*') {
        $reviewClass = 'uncited_probe_or_calibration_no_summary'
        $recommended = 'candidate_delete_after_log_read'
        $reason = 'Name indicates smoke/probe/calibration/short-budget run and no doc citations were found; retain logs/config if deleting weights.'
    } elseif ($hasTraceback) {
        $reviewClass = 'failed_or_interrupted_no_summary'
        $recommended = 'candidate_delete_after_log_read'
        $reason = 'remote_train.log contains failure/interruption markers and no summary exists.'
    } elseif ($cfg -eq $null) {
        $reviewClass = 'orphan_weight_no_config_no_summary'
        $recommended = 'keep_until_owner_review'
        $reason = 'No config and no summary; weight may be orphaned but needs owner review before deletion.'
    }

    $rows.Add([pscustomobject]@{
        exp_dir = $row.exp_dir
        exists = $true
        total_mb = $row.total_mb
        weight_count = $weights.Count
        weight_mb = $row.weight_mb
        weight_names = (($weights | ForEach-Object { $_.Name }) -join ';')
        summary_count = $summaries.Count
        config_exists = [bool](Test-Path $cfgPath)
        data_root = OneLine $cfg.data.data_root
        batch_size = SafeVal $cfg.training.batch_size
        num_epochs = SafeVal $cfg.training.num_epochs
        objective_mode = SafeVal $cfg.bridge.objective_mode
        loss_type = SafeVal $cfg.bridge.loss_type
        style_tokenizer = SafeVal $cfg.model.style_tokenizer
        style_spatial_mode = SafeVal $cfg.model.style_spatial_mode
        notes = OneLine $cfg.experiment.notes
        training_csv_count = $trainCsvs.Count
        latest_training_csv = if ($trainCsvs.Count -gt 0) { $trainCsvs[-1].FullName.Substring($Root.Length + 1) } else { '' }
        latest_training_tail = $latestTrainingTail
        remote_train_log_exists = [bool](Test-Path $remoteLog)
        remote_log_has_failure_marker = $hasTraceback
        remote_log_has_completion_marker = $hasComplete
        params_raw = $hasModelParams
        remote_train_head = $remoteHead
        remote_train_tail = $remoteTail
        top_level = ((Get-ChildItem -LiteralPath $dirPath -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -First 30 -ExpandProperty Name) -join ';')
        review_class = $reviewClass
        recommended_action = $recommended
        reason = $reason
    })
}

$rows | ConvertTo-Csv -NoTypeInformation
