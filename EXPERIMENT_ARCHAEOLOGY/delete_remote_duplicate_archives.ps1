$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = 'I:\Github\Latent_Style'

$targets = @(
    [pscustomobject]@{
        relative_path = 'eval_cache.zip'
        reason = 'stale redundant eval_cache archive; valid entries exist in eval_cache and only missing entry is invalid incomplete residue'
        retained_evidence = 'I:\Github\Latent_Style\eval_cache'
    },
    [pscustomobject]@{
        relative_path = 'Cycle-NCE\1-decoder-patch5-15_eAzEC.zip'
        reason = 'archive duplicates existing nonweight outputs and otherwise only preserves four old legacy epoch checkpoint weights'
        retained_evidence = 'I:\Github\Latent_Style\experiments\1-decoder-patch5-15 nonweight outputs'
    },
    [pscustomobject]@{
        relative_path = 'Cycle-NCE\src\45.rar'
        reason = 'exact SHA256 duplicate of Cycle-NCE\45.rar; root archive copy retained'
        retained_evidence = 'I:\Github\Latent_Style\Cycle-NCE\45.rar'
    }
)

$rootFull = [System.IO.Path]::GetFullPath($root)
$rows = New-Object System.Collections.Generic.List[object]

foreach ($target in $targets) {
    $path = Join-Path $root $target.relative_path
    $full = [System.IO.Path]::GetFullPath($path)
    if (-not $full.StartsWith($rootFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside root: $full"
    }

    $existsBefore = Test-Path -LiteralPath $full
    $beforeBytes = 0
    $beforeMb = 0
    $lastWrite = ''
    $status = 'missing_before_delete'
    $errorText = ''

    if ($existsBefore) {
        try {
            $item = Get-Item -LiteralPath $full -Force
            if ($item.PSIsContainer) {
                throw "Expected file but found directory: $full"
            }
            $beforeBytes = $item.Length
            $beforeMb = [math]::Round($item.Length / 1MB, 6)
            $lastWrite = [string]$item.LastWriteTime
            Remove-Item -LiteralPath $full -Force
            $status = 'deleted'
        } catch {
            $status = 'error'
            $errorText = $_.Exception.Message
        }
    }

    $postExists = Test-Path -LiteralPath $full
    if ($status -eq 'deleted' -and $postExists) {
        $status = 'post_delete_still_exists'
    }

    $rows.Add([pscustomobject]@{
        cleanup_run = 'remote_duplicate_archive_cleanup_20260605'
        remote_root = $root
        relative_path = $target.relative_path
        full_path = $full
        exists_before = $existsBefore
        before_bytes = $beforeBytes
        before_mb = $beforeMb
        last_write_time = $lastWrite
        status = $status
        post_exists = $postExists
        reason = $target.reason
        retained_evidence = $target.retained_evidence
        error = $errorText
    })
}

$rows | ConvertTo-Csv -NoTypeInformation
