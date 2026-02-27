echo "========================================="
echo "=== 正在执行极端消融实验: exp_1_control ==="
echo "========================================="
uv run run.py --config config_exp_1_control.json

echo "========================================="
echo "=== 正在执行极端消融实验: exp_2_zero_id ==="
echo "========================================="
uv run run.py --config config_exp_2_zero_id.json

echo "========================================="
echo "=== 正在执行极端消融实验: exp_3_macro_strokes ==="
echo "========================================="
uv run run.py --config config_exp_3_macro_strokes.json

echo "========================================="
echo "=== 正在执行极端消融实验: exp_4_zero_tv ==="
echo "========================================="
uv run run.py --config config_exp_4_zero_tv.json

echo "========================================="
echo "=== 正在执行极端消融实验: exp_5_signal_overdrive ==="
echo "========================================="
uv run run.py --config config_exp_5_signal_overdrive.json

echo "========================================="
echo "=== 正在执行极端消融实验: exp_6_nuclear ==="
echo "========================================="
uv run run.py --config config_exp_6_nuclear.json

