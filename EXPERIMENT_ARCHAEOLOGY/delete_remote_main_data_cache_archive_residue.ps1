$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = 'I:\Github\Latent_Style'

$targets = @(
    [pscustomobject]@{
        relative_path = 'eval_cache\hf\models--openai--clip-vit-base-patch32\blobs\a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f.incomplete'
        target_type = 'file'
        reason = 'failed_huggingface_clip_blob_incomplete'
    },
    [pscustomobject]@{
        relative_path = 'eval_cache\ref_feats_1558c2de70_m80.pt.tmp.575'
        target_type = 'file'
        reason = 'stale_ref_feats_tmp_residue'
    },
    [pscustomobject]@{
        relative_path = 'SchrodingerBridge\scale\datasets\wikiart_81k\.cache\huggingface\download\wCnio2FLeN37BG8ipDAohIO8E5U=.46096265c2d6143804b6bd1d07b5a59119f8b8f89e7a1d2cd4fca5592b450707.incomplete'
        target_type = 'file'
        reason = 'failed_wikiart81k_hf_download_incomplete'
    },
    [pscustomobject]@{
        relative_path = 'SchrodingerBridge\scale\datasets\wikiart_81k\.cache\huggingface\download\dataset.tar.gz.lock'
        target_type = 'file'
        reason = 'stale_wikiart81k_download_lock'
    },
    [pscustomobject]@{
        relative_path = 'Cycle-NCE\_archive\src_46\eval_cache\artfid\.cache\huggingface\download\qDTS-0xVe8KqtrFl6MozVPDSqJA=.5bc1c20380401911d99d528b7b6e4430044b56edd5f7f26467222f3fb7c54e7e.incomplete'
        target_type = 'file'
        reason = 'failed_artfid_download_incomplete_archive_copy'
    },
    [pscustomobject]@{
        relative_path = 'Cycle-NCE\_archive\src_46\eval_cache\artfid\.cache\huggingface\download\art_inception.pth.lock'
        target_type = 'file'
        reason = 'stale_artfid_download_lock_archive_copy'
    },
    [pscustomobject]@{
        relative_path = 'Cycle-NCE\src\eval_cache\artfid\.cache\huggingface\download\qDTS-0xVe8KqtrFl6MozVPDSqJA=.5bc1c20380401911d99d528b7b6e4430044b56edd5f7f26467222f3fb7c54e7e.incomplete'
        target_type = 'file'
        reason = 'failed_artfid_download_incomplete_src_copy'
    },
    [pscustomobject]@{
        relative_path = 'Cycle-NCE\src\eval_cache\artfid\.cache\huggingface\download\art_inception.pth.lock'
        target_type = 'file'
        reason = 'stale_artfid_download_lock_src_copy'
    },
    [pscustomobject]@{
        relative_path = 'Cycle-NCE\eval_cache\hf\models--openai--clip-vit-base-patch32\blobs\a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f.incomplete'
        target_type = 'file'
        reason = 'failed_cycle_nce_clip_blob_incomplete'
    },
    [pscustomobject]@{
        relative_path = 'experiments\eval_cache\hf\modelscope\stabilityai_sd-vae-ft-mse\._____temp'
        target_type = 'directory'
        reason = 'recursively_empty_modelscope_temp_dir'
    },
    [pscustomobject]@{
        relative_path = 'Related_Works\repos\S2WAT-main\pre_trained_models\tmp_timing'
        target_type = 'directory'
        reason = 'empty_tmp_timing_dir'
    }
)

function Get-TargetSizeBytes {
    param(
        [Parameter(Mandatory = $true)][string]$LiteralPath,
        [Parameter(Mandatory = $true)][string]$TargetType
    )

    if ($TargetType -eq 'file') {
        return (Get-Item -LiteralPath $LiteralPath -Force).Length
    }

    $files = @(Get-ChildItem -LiteralPath $LiteralPath -Recurse -File -Force -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) {
        return 0
    }
    return (($files | Measure-Object -Property Length -Sum).Sum)
}

$rootFull = [System.IO.Path]::GetFullPath($root)
$rows = New-Object System.Collections.Generic.List[object]

foreach ($target in $targets) {
    $path = Join-Path $root $target.relative_path
    $full = [System.IO.Path]::GetFullPath($path)
    $insideRoot = $full.StartsWith($rootFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)

    if (-not $insideRoot) {
        throw "Refusing path outside root: $full"
    }

    $existsBefore = Test-Path -LiteralPath $full
    $itemTypeBefore = ''
    $preSizeBytes = 0
    $preChildCount = ''
    $status = 'missing_before_delete'
    $note = ''

    if ($existsBefore) {
        $item = Get-Item -LiteralPath $full -Force
        $itemTypeBefore = if ($item.PSIsContainer) { 'directory' } else { 'file' }
        if ($itemTypeBefore -ne $target.target_type) {
            throw "Type mismatch for $full. Expected $($target.target_type), found $itemTypeBefore"
        }
        $preSizeBytes = Get-TargetSizeBytes -LiteralPath $full -TargetType $target.target_type
        if ($target.target_type -eq 'directory') {
            $preChildCount = @(Get-ChildItem -LiteralPath $full -Recurse -Force -ErrorAction SilentlyContinue).Count
        }

        Remove-Item -LiteralPath $full -Force -Recurse
        $status = 'deleted'
    }

    $postExists = Test-Path -LiteralPath $full
    if ($postExists) {
        $status = 'post_delete_still_exists'
        $note = 'verification_failed'
    }

    $rows.Add([pscustomobject]@{
        cleanup_run = 'remote_main_data_cache_archive_residue_20260605'
        remote_root = $root
        relative_path = $target.relative_path
        full_path = $full
        target_type = $target.target_type
        reason = $target.reason
        exists_before = $existsBefore
        item_type_before = $itemTypeBefore
        pre_size_bytes = $preSizeBytes
        pre_size_mb = [math]::Round($preSizeBytes / 1MB, 6)
        pre_child_count = $preChildCount
        status = $status
        post_exists = $postExists
        note = $note
    })
}

$rows | ConvertTo-Csv -NoTypeInformation
