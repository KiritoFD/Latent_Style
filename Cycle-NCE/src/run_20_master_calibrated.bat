echo "========================================="
echo "=== [START] 01_cap_64 | Perfect BS: 256 | LR: 0.000283 ==="
echo "========================================="
uv run run.py --config config_01_cap_64.json || echo "[WARNING] 01_cap_64 FAILED!"

echo "========================================="
echo "=== [START] 02_cap_128 | Perfect BS: 168 | LR: 0.000229 ==="
echo "========================================="
uv run run.py --config config_02_cap_128.json || echo "[WARNING] 02_cap_128 FAILED!"

echo "========================================="
echo "=== [START] 03_cap_192 | Perfect BS: 112 | LR: 0.000187 ==="
echo "========================================="
uv run run.py --config config_03_cap_192.json || echo "[WARNING] 03_cap_192 FAILED!"

echo "========================================="
echo "=== [START] 04_cap_256 | Perfect BS: 80 | LR: 0.000158 ==="
echo "========================================="
uv run run.py --config config_04_cap_256.json || echo "[WARNING] 04_cap_256 FAILED!"

echo "========================================="
echo "=== [START] 05_patch_micro | Perfect BS: 240 | LR: 0.000274 ==="
echo "========================================="
uv run run.py --config config_05_patch_micro.json || echo "[WARNING] 05_patch_micro FAILED!"

echo "========================================="
echo "=== [START] 06_patch_std | Perfect BS: 200 | LR: 0.000250 ==="
echo "========================================="
uv run run.py --config config_06_patch_std.json || echo "[WARNING] 06_patch_std FAILED!"

echo "========================================="
echo "=== [START] 07_patch_xmax | Perfect BS: 96 | LR: 0.000173 ==="
echo "========================================="
uv run run.py --config config_07_patch_xmax.json || echo "[WARNING] 07_patch_xmax FAILED!"

echo "========================================="
echo "=== [START] 08_lr_fast | Perfect BS: 168 | LR: 0.000500 ==="
echo "========================================="
uv run run.py --config config_08_lr_fast.json || echo "[WARNING] 08_lr_fast FAILED!"

echo "========================================="
echo "=== [START] 09_lr_slow | Perfect BS: 168 | LR: 0.000050 ==="
echo "========================================="
uv run run.py --config config_09_lr_slow.json || echo "[WARNING] 09_lr_slow FAILED!"

echo "========================================="
echo "=== [START] 10_wide_xmax | Perfect BS: 64 | LR: 0.000141 ==="
echo "========================================="
uv run run.py --config config_10_wide_xmax.json || echo "[WARNING] 10_wide_xmax FAILED!"

echo "========================================="
echo "=== [START] 11_narrow_xmax | Perfect BS: 120 | LR: 0.000194 ==="
echo "========================================="
uv run run.py --config config_11_narrow_xmax.json || echo "[WARNING] 11_narrow_xmax FAILED!"

echo "========================================="
echo "=== [START] 12_mid_xmax | Perfect BS: 72 | LR: 0.000150 ==="
echo "========================================="
uv run run.py --config config_12_mid_xmax.json || echo "[WARNING] 12_mid_xmax FAILED!"

echo "========================================="
echo "=== [START] 13_wide_micro | Perfect BS: 96 | LR: 0.000173 ==="
echo "========================================="
uv run run.py --config config_13_wide_micro.json || echo "[WARNING] 13_wide_micro FAILED!"

echo "========================================="
echo "=== [START] 14_narrow_micro | Perfect BS: 256 | LR: 0.000283 ==="
echo "========================================="
uv run run.py --config config_14_narrow_micro.json || echo "[WARNING] 14_narrow_micro FAILED!"

echo "========================================="
echo "=== [START] 15_split_brain_128 | Perfect BS: 96 | LR: 0.000173 ==="
echo "========================================="
uv run run.py --config config_15_split_brain_128.json || echo "[WARNING] 15_split_brain_128 FAILED!"

echo "========================================="
echo "=== [START] 16_split_brain_256 | Perfect BS: 64 | LR: 0.000141 ==="
echo "========================================="
uv run run.py --config config_16_split_brain_256.json || echo "[WARNING] 16_split_brain_256 FAILED!"

echo "========================================="
echo "=== [START] 17_extreme_underpowered | Perfect BS: 256 | LR: 0.000283 ==="
echo "========================================="
uv run run.py --config config_17_extreme_underpowered.json || echo "[WARNING] 17_extreme_underpowered FAILED!"

echo "========================================="
echo "=== [START] 18_extreme_overpowered | Perfect BS: 56 | LR: 0.000132 ==="
echo "========================================="
uv run run.py --config config_18_extreme_overpowered.json || echo "[WARNING] 18_extreme_overpowered FAILED!"

echo "========================================="
echo "=== [START] 19_the_abyss | Perfect BS: 72 | LR: 0.000150 ==="
echo "========================================="
uv run run.py --config config_19_the_abyss.json || echo "[WARNING] 19_the_abyss FAILED!"

echo "========================================="
echo "=== [START] 20_golden_balance | Perfect BS: 96 | LR: 0.000173 ==="
echo "========================================="
uv run run.py --config config_20_golden_balance.json || echo "[WARNING] 20_golden_balance FAILED!"

