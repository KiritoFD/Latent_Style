param(
    [string]$Root = 'I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$expRoot = Join-Path $Root 'exp'
$mediaExt = @('.png', '.jpg', '.jpeg', '.webp', '.gif')
$rows = New-Object System.Collections.Generic.List[object]

foreach ($dir in (Get-ChildItem -LiteralPath $expRoot -Directory | Sort-Object Name)) {
    $files = @(Get-ChildItem -LiteralPath $dir.FullName -Recurse -File -ErrorAction SilentlyContinue)
    $media = @($files | Where-Object { $_.Extension.ToLowerInvariant() -in $mediaExt })
    $generatedMedia = @($media | Where-Object { $_.FullName -match '\\generated\\' })
    $summaryGrid = @($media | Where-Object { $_.Name -like '*summary_grid*' -or $_.Name -like '*grid_first*' })
    $summaries = @($files | Where-Object { $_.Name -eq 'summary.json' })
    $csvs = @($files | Where-Object { $_.Extension.ToLowerInvariant() -eq '.csv' })
    $weights = @($files | Where-Object { $_.Extension.ToLowerInvariant() -in @('.pt', '.ckpt', '.pth') })

    $rows.Add([pscustomobject]@{
        remote_root = $Root
        exp_dir = $dir.Name
        total_files = $files.Count
        total_mb = ('{0:N3}' -f (($files | Measure-Object Length -Sum).Sum / 1MB))
        media_count = $media.Count
        media_mb = ('{0:N3}' -f (($media | Measure-Object Length -Sum).Sum / 1MB))
        generated_media_count = $generatedMedia.Count
        generated_media_mb = ('{0:N3}' -f (($generatedMedia | Measure-Object Length -Sum).Sum / 1MB))
        grid_media_count = $summaryGrid.Count
        grid_media_mb = ('{0:N3}' -f (($summaryGrid | Measure-Object Length -Sum).Sum / 1MB))
        summary_count = $summaries.Count
        csv_count = $csvs.Count
        weight_count = $weights.Count
        largest_media_examples = (($media | Sort-Object Length -Descending | Select-Object -First 8 | ForEach-Object {
            ('{0}:{1:N3}MB' -f $_.FullName.Substring($Root.Length + 1), ($_.Length / 1MB))
        }) -join ';')
    })
}

$rows | ConvertTo-Csv -NoTypeInformation
