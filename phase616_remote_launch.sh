#!/usr/bin/env bash
set -euo pipefail
cd ~/Latent_Style/SchrodingerBridge_phase616
python3 -c "import sys; sys.path.insert(0, 'src'); from run import load_experiment_config; from losses import OTFlowMatchingObjective; cfg=load_experiment_config('configs/aaai2027/phase616_ot_vertical_scratch_b8a2_e24.json'); obj=OTFlowMatchingObjective(cfg); print(obj.coupling_solver, obj.coupling_structure_cost_mode, obj.training_target_projection_mode)"
mkdir -p docs/experiments/phase2_fiber_bundle/616/logs/ot_vertical_round1
nohup env PYTHON_BIN=python3 bash tools/experiments/run_phase616_ot_vertical_round1.sh > docs/experiments/phase2_fiber_bundle/616/logs/ot_vertical_round1/launcher.log 2>&1 < /dev/null &
echo $! > docs/experiments/phase2_fiber_bundle/616/logs/ot_vertical_round1/launcher.pid
echo LAUNCHED_PID=$(cat docs/experiments/phase2_fiber_bundle/616/logs/ot_vertical_round1/launcher.pid)