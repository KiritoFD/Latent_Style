echo "========================================="
echo "Starting Experiment: A_baseline"
echo "========================================="
uv run run.py --config config_exp_A_baseline.json

echo "========================================="
echo "Starting Experiment: B_capacity"
echo "========================================="
uv run run.py --config config_exp_B_capacity.json

echo "========================================="
echo "Starting Experiment: C_macro_texture"
echo "========================================="
uv run run.py --config config_exp_C_macro_texture.json

echo "========================================="
echo "Starting Experiment: D_impasto_tv"
echo "========================================="
uv run run.py --config config_exp_D_impasto_tv.json

echo "========================================="
echo "Starting Experiment: E_safe_anchor"
echo "========================================="
uv run run.py --config config_exp_E_safe_anchor.json

