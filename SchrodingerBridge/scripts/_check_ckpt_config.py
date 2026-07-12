"""Check baseline checkpoint config."""
import torch, json

ckpt = torch.load(
    "I:/Github/Latent_Style/SchrodingerBridge/exp/refactor_clean_baseline/epoch_0005.pt",
    map_location="cpu",
    weights_only=False,
)
config = ckpt.get("config", {})
model = config.get("model", {})
bridge = config.get("bridge", {})
print("contract_family:", model.get("contract_family", "NOT_FOUND"))
print("solver_family:", model.get("solver_family", "NOT_FOUND"))
print("objective_mode:", bridge.get("objective_mode", "NOT_FOUND"))
print("transport_prediction_mode:", model.get("transport_prediction_mode", "NOT_FOUND"))
