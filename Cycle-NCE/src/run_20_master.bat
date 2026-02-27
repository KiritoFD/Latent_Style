echo "========================================="
echo "=== [START] Exp: 01_cap_64 | BS: 184 ==="
echo "========================================="
uv run run.py --config config_01_cap_64.json || echo "[WARNING] 01_cap_64 FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 02_cap_128 | BS: 88 ==="
echo "========================================="
uv run run.py --config config_02_cap_128.json || echo "[WARNING] 02_cap_128 FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 03_cap_192 | BS: 56 ==="
echo "========================================="
uv run run.py --config config_03_cap_192.json || echo "[WARNING] 03_cap_192 FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 04_cap_256 | BS: 40 ==="
echo "========================================="
uv run run.py --config config_04_cap_256.json || echo "[WARNING] 04_cap_256 FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 05_patch_micro | BS: 256 ==="
echo "========================================="
uv run run.py --config config_05_patch_micro.json || echo "[WARNING] 05_patch_micro FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 06_patch_std | BS: 152 ==="
echo "========================================="
uv run run.py --config config_06_patch_std.json || echo "[WARNING] 06_patch_std FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 07_patch_xmax | BS: 32 ==="
echo "========================================="
uv run run.py --config config_07_patch_xmax.json || echo "[WARNING] 07_patch_xmax FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 08_lr_fast | BS: 88 ==="
echo "========================================="
uv run run.py --config config_08_lr_fast.json || echo "[WARNING] 08_lr_fast FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 09_lr_slow | BS: 88 ==="
echo "========================================="
uv run run.py --config config_09_lr_slow.json || echo "[WARNING] 09_lr_slow FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 10_wide_xmax_slow | BS: 16 ==="
echo "========================================="
uv run run.py --config config_10_wide_xmax_slow.json || echo "[WARNING] 10_wide_xmax_slow FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 11_narrow_xmax_fast | BS: 64 ==="
echo "========================================="
uv run run.py --config config_11_narrow_xmax_fast.json || echo "[WARNING] 11_narrow_xmax_fast FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 12_mid_xmax_std | BS: 16 ==="
echo "========================================="
uv run run.py --config config_12_mid_xmax_std.json || echo "[WARNING] 12_mid_xmax_std FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 13_wide_micro_fast | BS: 152 ==="
echo "========================================="
uv run run.py --config config_13_wide_micro_fast.json || echo "[WARNING] 13_wide_micro_fast FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 14_narrow_micro_slow | BS: 256 ==="
echo "========================================="
uv run run.py --config config_14_narrow_micro_slow.json || echo "[WARNING] 14_narrow_micro_slow FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 15_split_brain_128 | BS: 32 ==="
echo "========================================="
uv run run.py --config config_15_split_brain_128.json || echo "[WARNING] 15_split_brain_128 FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 16_split_brain_256 | BS: 16 ==="
echo "========================================="
uv run run.py --config config_16_split_brain_256.json || echo "[WARNING] 16_split_brain_256 FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 17_extreme_underpowered | BS: 256 ==="
echo "========================================="
uv run run.py --config config_17_extreme_underpowered.json || echo "[WARNING] 17_extreme_underpowered FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 18_extreme_overpowered | BS: 96 ==="
echo "========================================="
uv run run.py --config config_18_extreme_overpowered.json || echo "[WARNING] 18_extreme_overpowered FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 19_the_abyss | BS: 16 ==="
echo "========================================="
uv run run.py --config config_19_the_abyss.json || echo "[WARNING] 19_the_abyss FAILED! Skipping..."

echo "========================================="
echo "=== [START] Exp: 20_golden_balance | BS: 40 ==="
echo "========================================="
uv run run.py --config config_20_golden_balance.json || echo "[WARNING] 20_golden_balance FAILED! Skipping..."

