$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = 'I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge'
$expRoot = Join-Path $root 'exp'
$searchRoots = @(
    (Join-Path $root 'docs'),
    (Join-Path $root 'aaai_submission')
) | Where-Object { Test-Path $_ }

$sourceFiles = New-Object System.Collections.Generic.List[object]
foreach ($searchRoot in $searchRoots) {
    Get-ChildItem -LiteralPath $searchRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in '.md', '.csv', '.txt', '.json', '.tex' } |
        ForEach-Object { $sourceFiles.Add($_) }
}

$rows = New-Object System.Collections.Generic.List[object]
foreach ($dir in (Get-ChildItem -LiteralPath $expRoot -Directory | Sort-Object Name)) {
    $hits = New-Object System.Collections.Generic.List[string]
    foreach ($file in $sourceFiles) {
        try {
            $match = Select-String -LiteralPath $file.FullName -SimpleMatch -Pattern $dir.Name -Quiet
            if ($match) {
                $hits.Add($file.FullName.Substring($root.Length + 1))
            }
        } catch {
            # Ignore unreadable text-like files; binary artifacts are excluded above.
        }
    }

    $masterHits = @($hits | Where-Object { $_ -like '*aaai2027_master_experiment_log.csv' })
    $reviewHits = @($hits | Where-Object { $_ -like 'docs\reviews\*' })
    $experimentHits = @($hits | Where-Object { $_ -like 'docs\experiments\*' })
    $paperHits = @($hits | Where-Object { $_ -like 'aaai_submission\*' })

    $rows.Add([pscustomobject]@{
        remote_root = $root
        exp_dir = $dir.Name
        source_file_count = $sourceFiles.Count
        total_hit_count = $hits.Count
        master_log_hit_count = $masterHits.Count
        experiment_doc_hit_count = $experimentHits.Count
        review_doc_hit_count = $reviewHits.Count
        paper_hit_count = $paperHits.Count
        hit_files = ($hits | Sort-Object -Unique) -join ';'
    })
}

$rows | ConvertTo-Csv -NoTypeInformation
