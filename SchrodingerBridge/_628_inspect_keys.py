import json
p = r"I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive\X1_velmag_w10\full_eval\epoch_0010\summary.json"
d = json.load(open(p, "r", encoding="utf-8"))
a = d.get("analysis", {})
print("analysis keys:", list(a.keys()))
for k, v in a.items():
    print(f"\n=== {k} === type={type(v).__name__}")
    if isinstance(v, dict):
        print(f"  subkeys: {list(v.keys())[:15]}")
        for sk, sv in list(v.items())[:5]:
            print(f"    {sk}: {type(sv).__name__} = {sv if not isinstance(sv,(dict,list)) else '...'}")
