Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$evalDir = "exp\710_b0_weave\full_eval\epoch_0010"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"

# Run DINO metrics
$dinoStart = Get-Date
python -u src\utils\compute_dino_metrics.py `
    --eval_dir $evalDir `
    --test_dir $testDir `
    --batch_size 4 --max_refs_per_style 30 `
    --allow_network `
    *> exp\710_b0_weave_dino_log.txt
$dinoMin = [math]::Round(((Get-Date) - $dinoStart).TotalMinutes, 1)
Write-Host "DINO: ${dinoMin}min"

# Extract metrics
python -c "import csv; rows=list(csv.DictReader(open(r'exp\710_b0_weave\full_eval\epoch_0010\metrics.csv',encoding='utf-8'))); dino=list(csv.DictReader(open(r'exp\710_b0_weave\full_eval\epoch_0010\dino_metrics.csv',encoding='utf-8'))); n=len(rows); cs=sum(float(r['clip_style']) for r in rows)/n; lp=sum(float(r['content_lpips']) for r in rows)/n; ds=sum(float(r['dino_s']) for r in dino)/n; dc=sum(float(r['dino_c']) for r in dino)/n; print(f'n={n} all_clip={cs:.4f} all_lpips={lp:.4f} all_dino_s={ds:.4f} all_dino_c={dc:.4f}')"
