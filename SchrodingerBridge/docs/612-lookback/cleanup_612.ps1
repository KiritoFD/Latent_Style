# SchrodingerBridge 清理脚本 — 2026-06-12 回顾后生成
# 
# 说明: 此脚本列出可安全删除的文件。实际删除前请务必确认。
# 本地: G:\GitHub\Latent_Style\SchrodingerBridge\
# 远程: I:\GitHub\Latent_Style\SchrodingerBridge\ (ssh administrator@100.115.18.62 -p 2222)
#
# 执行本地清理:
#   powershell -ExecutionPolicy Bypass -File cleanup_612.ps1 -WhatIf
# 实际执行:
#   powershell -ExecutionPolicy Bypass -File cleanup_612.ps1

$LocalRoot = "G:\GitHub\Latent_Style\SchrodingerBridge"

# ============================================================
# 1. round1_attn_sa_mod 中间 checkpoint (保留最后 epoch_0024)
#    ~4-5 GB 可回收
# ============================================================
$RemovePaths = @(
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0001.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0002.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0003.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0004.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0005.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0006.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0007.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0008.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0009.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0010.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0011.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0012.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0013.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0014.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0015.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0016.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0017.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0018.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0019.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0020.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0021.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0022.pt",
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0023.pt",

    # 重复文件
    "$LocalRoot\aaai2027\round1_attn_sa_mod_fast_local\checkpoints\epoch_0001.pt.pt",

    # aaai2027 根目录陈旧 checkpoint
    "$LocalRoot\aaai2027\carriergate_fresh_epoch_0002.pt",
    "$LocalRoot\aaai2027\hold4twostage_epoch_0002.pt",
    "$LocalRoot\aaai2027\knee_carriergate_fresh_epoch_0002.pt",
    "$LocalRoot\aaai2027\knee_carriergate_fresh_epoch_0012.pt"
)

# ============================================================
# 2. S-add 实验中间 epoch
# ============================================================
$SAddDir = "$LocalRoot\S-add__K-1_C-0_W-20_Col-0\full_eval"
$RemoveDirs = @(
    # 保留 epoch_0008 (最后), 删除 1-7
    "$SAddDir\epoch_0001",
    "$SAddDir\epoch_0002",
    "$SAddDir\epoch_0003", 
    "$SAddDir\epoch_0004",
    "$SAddDir\epoch_0005",
    "$SAddDir\epoch_0006",
    "$SAddDir\epoch_0007",

    # 中间 sweep 目录
    "$LocalRoot\S-add__K-1_C-0_W-20_Col-0\step_size_sweep_epoch7",
    "$LocalRoot\S-add__K-1_C-0_W-20_Col-0\residual_scale_sweep_epoch7",
    "$LocalRoot\S-add__K-1_C-0_W-20_Col-0\full_eval_timing_epoch7"
)

# ============================================================
# 3. 大体积 archive 目录
# ============================================================
$RemoveArchiveDirs = @(
    "$LocalRoot\archives\exp_archive_20260526_051536",
    "$LocalRoot\archives\old_experiment_dirs",
    "$LocalRoot\archives\old_paper_workspaces",
    "$LocalRoot\archives\old_root_files",
    "$LocalRoot\archives\code_backups"
)

# ============================================================
# 4. _codex_tmp 工具脚本和日志
# ============================================================
$RemoveCodexDirs = @(
    "$LocalRoot\_codex_tmp\aaai27_authorkit",
    "$LocalRoot\_codex_tmp\distinct5_passthrough_test",
    "$LocalRoot\_codex_tmp\remote_fetch_test"
)

# ============================================================
# 5. 临时文件 (tar, zip)
# ============================================================
$RemoveTempFiles = @(
    "$LocalRoot\_codex_tmp\sd_turbo_snapshot_20260606.tar",
    "$LocalRoot\_codex_tmp\split2_post_impressionism_patch.tar",
    "$LocalRoot\_codex_tmp\split3_missing_styles_patch.tar"
)

# ============================================================
# 执行
# ============================================================
$TotalRemoved = 0
foreach ($P in $RemovePaths) {
    if (Test-Path -LiteralPath $P) {
        $size = (Get-Item -LiteralPath $P).Length
        $sizeMB = [math]::Round($size / 1MB, 1)
        Write-Host "[FILE] $P ($sizeMB MB)" -ForegroundColor Yellow
        Remove-Item -LiteralPath $P -Force
        $TotalRemoved += $size
    } else {
        Write-Host "[SKIP] $P (not found)" -ForegroundColor DarkGray
    }
}
foreach ($D in $RemoveDirs + $RemoveArchiveDirs + $RemoveCodexDirs) {
    if (Test-Path -LiteralPath $D) {
        $size = (Get-ChildItem -LiteralPath $D -Recurse -File | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size / 1MB, 1)
        Write-Host "[DIR]  $D ($sizeMB MB)" -ForegroundColor Red
        Remove-Item -LiteralPath $D -Recurse -Force
        $TotalRemoved += $size
    } else {
        Write-Host "[SKIP] $D (not found)" -ForegroundColor DarkGray
    }
}
foreach ($F in $RemoveTempFiles) {
    if (Test-Path -LiteralPath $F) {
        $size = (Get-Item -LiteralPath $F).Length
        $sizeMB = [math]::Round($size / 1MB, 1)
        Write-Host "[TEMP] $F ($sizeMB MB)" -ForegroundColor Magenta
        Remove-Item -LiteralPath $F -Force
        $TotalRemoved += $size
    }
}
$totalMB = [math]::Round($TotalRemoved / 1MB, 1)
Write-Host "`nTotal freed: $totalMB MB" -ForegroundColor Green
