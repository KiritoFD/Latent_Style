import json
d = json.load(open('/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/t1_asg_5ep/full_eval/epoch_0005/summary.json'))
t = d.get('timings_sec', {})
print("=== INFERENCE TIMINGS_SEC ===")
print(json.dumps(t, indent=2, default=str))
s = d.get('settings', {})
print("\n=== INFERENCE SETTINGS ===")
print(json.dumps(s, indent=2, default=str))
