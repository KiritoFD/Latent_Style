import json
import sys

path = "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_phase2_smoe_fiber_sde_fiberwise_swd_k070/full_eval/epoch_0001/summary.json"
try:
    with open(path) as f:
        data = json.load(f)
    print("ALL PAIRS OVERVIEW:")
    print(json.dumps(data["analysis"]["all_pairs_overview"], indent=2))
except Exception as e:
    print("Error:", e)
