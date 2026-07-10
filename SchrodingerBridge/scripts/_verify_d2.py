"""Verify D2 config loads correctly and check moment loss is wired."""
import sys
import os
ROOT = r"I:\Github\Latent_Style\SchrodingerBridge"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from config_schema import ExperimentConfig

c = ExperimentConfig.load("configs/d2_moment_hf1_15ep.json")
print("config loaded OK")
print("bridge.w_moment_hf =", c.bridge.w_moment_hf)
print("bridge.w_moment_ll =", c.bridge.w_moment_ll)
print("bridge.w_gram_hf =", getattr(c.bridge, "w_gram_hf", 0.0))
print("training.batch_size =", c.training.batch_size)
print("checkpoint.save_dir =", c.checkpoint.save_dir)

# Quick import test
from src.spectral_losses620 import SpectralODEObjective620
print("spectral_losses620 imported OK")
print("has _moment_loss:", hasattr(SpectralODEObjective620, "_moment_loss"))
print("has _gram_loss:", hasattr(SpectralODEObjective620, "_gram_loss"))
