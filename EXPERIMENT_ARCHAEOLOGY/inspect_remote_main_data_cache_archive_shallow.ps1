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

function JoinText($Items) {
    if ($null -eq $Items) { return '' }
    return (($Items | ForEach-Object { $_ }) -join ';')
}

$rows = New-Object System.Collections.Generic.List[object]

foreach ($rel in $paths) {
    $path = Join-Path $root $rel
    if (!(Test-Path $path)) {
        $rows.Add([pscustomobject]@{
            remote_root = $root
            relative_path = $rel
            exists = $false
            immediate_dirs = ''
            immediate_files = ''
            sample_recursive_files = ''
            largest_sample_files = ''
            incomplete_candidates = ''
            empty_temp_dirs = ''
            note = 'missing'
        })
        continue
    }

    $children = @(Get-ChildItem -LiteralPath $path -Force -ErrorAction SilentlyContinue)
    $immediateDirs = @($children | Where-Object { $_.PSIsContainer } | Sort-Object Name | Select-Object -First 60 | ForEach-Object { $_.Name })
    $immediateFiles = @($children | Where-Object { !$_.PSIsContainer } | Sort-Object Name | Select-Object -First 60 | ForEach-Object {
        ('{0}:{1:N3}MB' -f $_.Name, ($_.Length / 1MB))
    })

    $sampleFiles = @(Get-ChildItem -LiteralPath $path -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object FullName | Select-Object -First 30 | ForEach-Object { $_.FullName.Substring($root.Length + 1) })
    $largestFiles = @(Get-ChildItem -LiteralPath $path -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object Length -Descending | Select-Object -First 20 | ForEach-Object {
            ('{0}:{1:N3}MB' -f $_.FullName.Substring($root.Length + 1), ($_.Length / 1MB))
        })
    $badCandidates = @(Get-ChildItem -LiteralPath $path -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like '*.incomplete' -or $_.Name -like '*.tmp' -or $_.Extension.ToLowerInvariant() -eq '.tmp' } |
        Sort-Object Length -Descending | Select-Object -First 50 | ForEach-Object {
            ('{0}:{1:N3}MB' -f $_.FullName.Substring($root.Length + 1), ($_.Length / 1MB))
        })
    $emptyTempDirs = @(Get-ChildItem -LiteralPath $path -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'temp|tmp|\._____temp|__pycache__' } |
        Where-Object { @(Get-ChildItem -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue).Count -eq 0 } |
        Sort-Object FullName | Select-Object -First 50 | ForEach-Object { $_.FullName.Substring($root.Length + 1) })

    $rows.Add([pscustomobject]@{
        remote_root = $root
        relative_path = $rel
        exists = $true
        immediate_dirs = JoinText $immediateDirs
        immediate_files = JoinText $immediateFiles
        sample_recursive_files = JoinText $sampleFiles
        largest_sample_files = JoinText $largestFiles
        incomplete_candidates = JoinText $badCandidates
        empty_temp_dirs = JoinText $emptyTempDirs
        note = 'shallow opened plus recursive samples/bad-cache candidates'
    })
}

$rows | ConvertTo-Csv -NoTypeInformation
