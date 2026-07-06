# Master pipeline: run all wikiarts-20 evaluations sequentially
#   1. baselines (identity/adain/wct): generate distinct5 part + evaluate 20 styles
#   2. WD-VF (random-20 model): evaluate 20 styles
#
# Launched via schtasks so it survives ssh disconnect.

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$LOG = "$REPO\logs\wikiarts20_all.log"
$PYTHON = "C:\Program Files\Python312\python.exe"

New-Item -ItemType Directory -Force -Path "$REPO\logs" | Out-Null

"=== wikiarts-20 ALL pipeline started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append

# ── Step 1: baselines (identity/adain/wct) ──
"--- Step 1: baselines (identity/adain/wct) ---" | Tee-Object -FilePath $LOG -Append
$baselinesScript = "$REPO\scripts\_eval_baselines_wikiarts20.ps1"
if (Test-Path $baselinesScript) {
    & powershell -ExecutionPolicy Bypass -File $baselinesScript
    "  baselines script exit code: $LASTEXITCODE" | Tee-Object -FilePath $LOG -Append
} else {
    "  ERROR: baselines script not found: $baselinesScript" | Tee-Object -FilePath $LOG -Append
}

# ── Step 2: WD-VF eval ──
"--- Step 2: WD-VF eval ---" | Tee-Object -FilePath $LOG -Append
$wdvfScript = "$REPO\scripts\_eval_wikiarts20.ps1"
if (Test-Path $wdvfScript) {
    & powershell -ExecutionPolicy Bypass -File $wdvfScript
    "  WD-VF script exit code: $LASTEXITCODE" | Tee-Object -FilePath $LOG -Append
} else {
    "  ERROR: WD-VF script not found: $wdvfScript" | Tee-Object -FilePath $LOG -Append
}

"=== wikiarts-20 ALL pipeline finished at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append

# ── Final summary: aggregate metrics.csv for each method ──
"--- Final summary ---" | Tee-Object -FilePath $LOG -Append

$aggScript = @"
import csv
from pathlib import Path

OUT_ROOT = Path(r"$REPO\exp\baseline_wikiarts20")
EV_DIR = Path(r"$REPO\exp\wikiarts20_eval")

def agg(csv_path):
    if not csv_path.exists():
        return None, None, 0
    clip_s_sum, lpips_sum, n = 0.0, 0.0, 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                lpips = float(row[3])  # col 4 (0-indexed 3) = content_lpips
                clip_s = float(row[5])  # col 6 (0-indexed 5) = clip_style
                clip_s_sum += clip_s
                lpips_sum += lpips
                n += 1
            except (ValueError, IndexError):
                continue
    if n == 0:
        return None, None, 0
    return clip_s_sum / n, lpips_sum / n, n

for method in ['identity', 'adain', 'wct']:
    csv_path = OUT_ROOT / method / 'metrics.csv'
    cs, lp, n = agg(csv_path)
    if n > 0:
        print(f'{method:12s}  CLIP-S={cs:.4f}  LPIPS={lp:.4f}  n={n}')
    else:
        print(f'{method:12s}  NO METRICS')

cs, lp, n = agg(EV_DIR / 'metrics.csv')
if n > 0:
    print(f'{"WD-VF":12s}  CLIP-S={cs:.4f}  LPIPS={lp:.4f}  n={n}')
else:
    print(f'{"WD-VF":12s}  NO METRICS')
"@

$aggPath = "$REPO\scripts\_agg_wikiarts20_final.py"
$aggScript | Out-File -FilePath $aggPath -Encoding utf8 -Force

"  aggregating metrics..." | Tee-Object -FilePath $LOG -Append
$aggOut = & $PYTHON $aggPath 2>&1
$aggOut | ForEach-Object { "  $_" } | Tee-Object -FilePath $LOG -Append
