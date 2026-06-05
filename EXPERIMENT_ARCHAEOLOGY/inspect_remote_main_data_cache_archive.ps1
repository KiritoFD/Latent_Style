$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = 'I:\Github\Latent_Style'
$paths = @(
    'data',
    'style_data',
    'latents',
    'latents_overfit50',
    'latent-256',
    'latent-256-flux1',
    'latent-256-flux2',
    'latent-256-kl-f4',
    'latent-256-kl-f4-mode',
    'latent-256-sd15-ema',
    'latent-256-sdxl',
    'latent-256-sdxl-fp32',
    'eval_cache',
    'SchrodingerBridge\scale\datasets',
    'SchrodingerBridge\S-add__K-1_C-0_W-20_Col-0',
    'SchrodingerBridge\review_additional_experiments',
    'Cycle-NCE',
    'experiments',
    'StarGAN',
    'seedream45_api',
    'Related_Works\baseline_pipeline\results',
    'Related_Works\repos'
)

$weightExt = @('.pt', '.pth', '.ckpt', '.safetensors')
$mediaExt = @('.png', '.jpg', '.jpeg', '.webp', '.gif')
$archiveExt = @('.zip', '.tar', '.gz', '.tgz', '.rar', '.7z')

function SizeMb($Files) {
    if ($null -eq $Files -or $Files.Count -eq 0) { return '0.000' }
    return ('{0:N3}' -f (($Files | Measure-Object Length -Sum).Sum / 1MB))
}

function JoinNames($Items) {
    if ($null -eq $Items) { return '' }
    return (($Items | ForEach-Object { $_ }) -join ';')
}

$rows = New-Object System.Collections.Generic.List[object]

foreach ($rel in $paths) {
    $path = Join-Path $root $rel
    if (!(Test-Path $path)) {
        $rows.Add([pscustomobject]@{
            remote_root = $root
            path = $path
            relative_path = $rel
            exists = $false
            file_count = 0
            dir_count = 0
            total_mb = '0.000'
            weight_count = 0
            weight_mb = '0.000'
            media_count = 0
            media_mb = '0.000'
            archive_count = 0
            archive_mb = '0.000'
            incomplete_count = 0
            incomplete_mb = '0.000'
            empty_dir_count = 0
            temp_dir_count = 0
            top_children_by_size = ''
            largest_files = ''
            sample_files = ''
        })
        continue
    }

    $dirs = @(Get-ChildItem -LiteralPath $path -Recurse -Directory -ErrorAction SilentlyContinue)
    $files = @(Get-ChildItem -LiteralPath $path -Recurse -File -ErrorAction SilentlyContinue)
    $weights = @($files | Where-Object { $_.Extension.ToLowerInvariant() -in $weightExt })
    $media = @($files | Where-Object { $_.Extension.ToLowerInvariant() -in $mediaExt })
    $archives = @($files | Where-Object { $_.Extension.ToLowerInvariant() -in $archiveExt -or $_.Name -match '\.tar\.' })
    $incomplete = @($files | Where-Object { $_.Name -like '*.incomplete' -or $_.Name -like '*.tmp' -or $_.Extension.ToLowerInvariant() -eq '.tmp' })
    $emptyDirs = @($dirs | Where-Object { @(Get-ChildItem -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue).Count -eq 0 })
    $tempDirs = @($dirs | Where-Object { $_.Name -match 'temp|tmp|__pycache__|\.cache|\._____temp' })

    $children = @(Get-ChildItem -LiteralPath $path -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.PSIsContainer) {
            $childFiles = @(Get-ChildItem -LiteralPath $_.FullName -Recurse -File -ErrorAction SilentlyContinue)
            [pscustomobject]@{ name = $_.Name; mb = [double](($childFiles | Measure-Object Length -Sum).Sum / 1MB); files = $childFiles.Count }
        } else {
            [pscustomobject]@{ name = $_.Name; mb = [double]($_.Length / 1MB); files = 1 }
        }
    } | Sort-Object mb -Descending | Select-Object -First 12 | ForEach-Object {
        ('{0}:{1:N3}MB:{2}files' -f $_.name, $_.mb, $_.files)
    })

    $largest = @($files | Sort-Object Length -Descending | Select-Object -First 12 | ForEach-Object {
        ('{0}:{1:N3}MB' -f $_.FullName.Substring($root.Length + 1), ($_.Length / 1MB))
    })

    $samples = @($files | Sort-Object FullName | Select-Object -First 12 | ForEach-Object {
        $_.FullName.Substring($root.Length + 1)
    })

    $rows.Add([pscustomobject]@{
        remote_root = $root
        path = $path
        relative_path = $rel
        exists = $true
        file_count = $files.Count
        dir_count = $dirs.Count
        total_mb = SizeMb $files
        weight_count = $weights.Count
        weight_mb = SizeMb $weights
        media_count = $media.Count
        media_mb = SizeMb $media
        archive_count = $archives.Count
        archive_mb = SizeMb $archives
        incomplete_count = $incomplete.Count
        incomplete_mb = SizeMb $incomplete
        empty_dir_count = $emptyDirs.Count
        temp_dir_count = $tempDirs.Count
        top_children_by_size = JoinNames $children
        largest_files = JoinNames $largest
        sample_files = JoinNames $samples
    })
}

$rows | ConvertTo-Csv -NoTypeInformation
