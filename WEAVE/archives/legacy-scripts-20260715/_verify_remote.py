import sys
sys.path.insert(0, ".")
from flow import FlowMatchingObjective
from config_schema import load_experiment_config
cfg = load_experiment_config("default_config.json")
obj = FlowMatchingObjective(cfg)
print("REMOTE OK")
print("  epochs=", cfg.training.num_epochs, "save_interval=", cfg.training.save_interval, "each_epoch=", cfg.training.full_eval_each_epoch)
print("  w_ll=", obj.w_ll, "alpha=", obj.ll_partial_alpha, "mode=", obj.ll_partial_mode)
print("  bridge_sigma=", obj.bridge_sigma, "SAT=", obj.structure_aligned_target)
print("  save_dir=", cfg.checkpoint.save_dir)
