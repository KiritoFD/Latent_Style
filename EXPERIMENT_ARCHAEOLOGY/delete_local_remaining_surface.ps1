Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$CleanupDir = Join-Path $PSScriptRoot 'cleanup'
$LedgerPath = Join-Path $CleanupDir 'manual_local_remaining_surface_cleanup_20260605.csv'
$VerifyPath = Join-Path $PSScriptRoot 'manual_local_remaining_surface_post_delete_verify_20260605.csv'

if (-not (Test-Path -LiteralPath $CleanupDir)) {
    New-Item -ItemType Directory -Path $CleanupDir | Out-Null
}

$targets = @(
    [pscustomobject]@{
        RelativePath = 'eval_cache\vae_onnx\ema_b2_64\trt_cache'
        TargetType = 'directory'
        CleanupClass = 'empty_cache_directory'
        Evidence = 'recursive child count 0 before deletion'
    },
    [pscustomobject]@{
        RelativePath = 'SchrodingerBridge\exp\frontier\decision_tree_clip_style\s21_temp_var0p0_temp0p03'
        TargetType = 'directory'
        CleanupClass = 'empty_probe_directory'
        Evidence = 'recursive child count 0 before deletion'
    },
    [pscustomobject]@{
        RelativePath = 'SchrodingerBridge\datasets\horse2zebra\raw\horse2zebra.zip'
        TargetType = 'file'
        CleanupClass = 'fully_duplicated_dataset_zip'
        Evidence = 'zip entries 2661; missing same-size extracted entries 0'
    },
    [pscustomobject]@{
        RelativePath = 'Related_Works\runs\cut_5x5\cut.zip'
        TargetType = 'file'
        CleanupClass = 'fully_duplicated_result_zip'
        Evidence = 'zip entries 2520; missing same-size extracted entries 0'
    },
    [pscustomobject]@{
        RelativePath = 'exp\highres_eval_local\samst_outputs_epoch50.tar'
        TargetType = 'file'
        CleanupClass = 'fully_duplicated_output_tar'
        Evidence = 'tar file entries 750; missing or mismatched extracted files 0'
    }
)

function Resolve-ExistingInRepo {
    param([string]$RelativePath)
    $candidate = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $candidate)) {
        return $null
    }
    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    $prefix = $RepoRoot.TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside repo root: $resolved"
    }
    return $resolved
}

function Get-TargetSizeInfo {
    param([string]$Path)
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.PSIsContainer) {
        $children = @(Get-ChildItem -LiteralPath $Path -Force -Recurse -ErrorAction SilentlyContinue)
        $files = @($children | Where-Object { -not $_.PSIsContainer })
        $bytes = 0
        if ($files.Count -gt 0) {
            $measure = $files | Measure-Object -Property Length -Sum
            if ($null -ne $measure.Sum) { $bytes = $measure.Sum }
        }
        return [pscustomobject]@{
            IsDirectory = $true
            FileCount = $files.Count
            ChildCount = $children.Count
            SizeMB = [math]::Round([double]$bytes / 1MB, 6)
        }
    }
    return [pscustomobject]@{
        IsDirectory = $false
        FileCount = 1
        ChildCount = 0
        SizeMB = [math]::Round([double]$item.Length / 1MB, 6)
    }
}

$ledger = foreach ($target in $targets) {
    $resolved = Resolve-ExistingInRepo -RelativePath $target.RelativePath
    $existsBefore = $null -ne $resolved
    $deleted = $false
    $note = ''
    $sizeInfo = [pscustomobject]@{ IsDirectory = ''; FileCount = ''; ChildCount = ''; SizeMB = '' }

    if ($existsBefore) {
        $sizeInfo = Get-TargetSizeInfo -Path $resolved
        if ($target.TargetType -eq 'directory' -and -not $sizeInfo.IsDirectory) {
            throw "Whitelist expected directory but found file: $($target.RelativePath)"
        }
        if ($target.TargetType -eq 'file' -and $sizeInfo.IsDirectory) {
            throw "Whitelist expected file but found directory: $($target.RelativePath)"
        }
        if ($target.TargetType -eq 'directory' -and $sizeInfo.ChildCount -ne 0) {
            $note = "Skipped: directory no longer empty; child_count=$($sizeInfo.ChildCount)"
        } else {
            Remove-Item -LiteralPath $resolved -Force -Recurse
            $deleted = -not (Test-Path -LiteralPath $resolved)
            if (-not $deleted) {
                throw "Deletion failed: $($target.RelativePath)"
            }
            $note = 'Deleted exact whitelist target'
        }
    } else {
        $note = 'Already absent before this pass'
    }

    [pscustomobject]@{
        timestamp = (Get-Date).ToString('s')
        scope = 'local_remaining_surface'
        relative_path = $target.RelativePath
        target_type = $target.TargetType
        cleanup_class = $target.CleanupClass
        existed_before = $existsBefore
        size_mb_before = $sizeInfo.SizeMB
        file_count_before = $sizeInfo.FileCount
        child_count_before = $sizeInfo.ChildCount
        evidence = $target.Evidence
        deleted = $deleted
        exists_after = if ($existsBefore) { Test-Path -LiteralPath (Join-Path $RepoRoot $target.RelativePath) } else { $false }
        note = $note
    }
}

$ledger | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $LedgerPath

$verifyTargets = @(
    [pscustomobject]@{ RelativePath = 'eval_cache\vae_onnx\ema_b2_64\trt_cache'; Expected = 'absent'; Reason = 'deleted empty cache dir' },
    [pscustomobject]@{ RelativePath = 'SchrodingerBridge\exp\frontier\decision_tree_clip_style\s21_temp_var0p0_temp0p03'; Expected = 'absent'; Reason = 'deleted empty probe dir' },
    [pscustomobject]@{ RelativePath = 'SchrodingerBridge\datasets\horse2zebra\raw\horse2zebra.zip'; Expected = 'absent'; Reason = 'deleted fully duplicated zip' },
    [pscustomobject]@{ RelativePath = 'SchrodingerBridge\datasets\horse2zebra\raw\horse2zebra\trainA'; Expected = 'present'; Reason = 'retained extracted horse2zebra trainA evidence' },
    [pscustomobject]@{ RelativePath = 'SchrodingerBridge\datasets\horse2zebra\raw\horse2zebra\testA'; Expected = 'present'; Reason = 'retained extracted horse2zebra testA evidence' },
    [pscustomobject]@{ RelativePath = 'Related_Works\runs\cut_5x5\cut.zip'; Expected = 'absent'; Reason = 'deleted fully duplicated zip' },
    [pscustomobject]@{ RelativePath = 'Related_Works\runs\cut_5x5\infer_5x5'; Expected = 'present'; Reason = 'retained extracted cut result tree' },
    [pscustomobject]@{ RelativePath = 'exp\highres_eval_local\samst_outputs_epoch50.tar'; Expected = 'absent'; Reason = 'deleted fully duplicated output tar' },
    [pscustomobject]@{ RelativePath = 'exp\highres_eval_local\samst\Baroque\outputs\epoch_50'; Expected = 'present'; Reason = 'retained extracted SaMST output tree' },
    [pscustomobject]@{ RelativePath = 'exp\highres_eval_local\samst_ckpts_epoch50.tar'; Expected = 'present'; Reason = 'retained non-duplicated checkpoint archive' },
    [pscustomobject]@{ RelativePath = 'Related_Works\runs\lbm_train_wds_smoke_photo_to_monet\train-000000.tar'; Expected = 'present'; Reason = 'retained WebDataset shard' },
    [pscustomobject]@{ RelativePath = 'Related_Works\runs\lbm_train_wds_smoke_photo_to_monet\val-000000.tar'; Expected = 'present'; Reason = 'retained WebDataset shard' },
    [pscustomobject]@{ RelativePath = 'Related_Works\repos\ArtBank\clip\bpe_simple_vocab_16e6.txt.gz'; Expected = 'present'; Reason = 'retained dependency gzip' },
    [pscustomobject]@{ RelativePath = 'Cycle-NCE\uv.lock'; Expected = 'present'; Reason = 'retained uv dependency lock' },
    [pscustomobject]@{ RelativePath = 'Cycle-NCE\src\uv.lock'; Expected = 'present'; Reason = 'retained uv dependency lock' }
)

$verify = foreach ($target in $verifyTargets) {
    $path = Join-Path $RepoRoot $target.RelativePath
    $exists = Test-Path -LiteralPath $path
    $fileCount = ''
    $sizeMB = ''
    if ($exists) {
        $info = Get-TargetSizeInfo -Path $path
        $fileCount = $info.FileCount
        $sizeMB = $info.SizeMB
    }
    [pscustomobject]@{
        timestamp = (Get-Date).ToString('s')
        relative_path = $target.RelativePath
        expected = $target.Expected
        exists = $exists
        file_count = $fileCount
        size_mb = $sizeMB
        pass = (($target.Expected -eq 'present' -and $exists) -or ($target.Expected -eq 'absent' -and -not $exists))
        reason = $target.Reason
    }
}

$verify | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $VerifyPath
$ledger | Format-Table -AutoSize
