import json
import sys

path = "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/decision_tree_highpass_run/full_eval/epoch_0001/summary.json"
try:
    with open(path) as f:
        data = json.load(f)
    print("ALL PAIRS OVERVIEW:")
    print(json.dumps(data["analysis"]["all_pairs_overview"], indent=2))
except Exception as e:
    print("Error:", e)
