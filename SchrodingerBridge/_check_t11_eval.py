import json
summary_path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\full_eval\epoch_0001\summary.json"
s = json.load(open(summary_path))
# Check matrix_breakdown
mb = s.get("matrix_breakdown", {})
print("matrix_breakdown keys:", list(mb.keys())[:10] if isinstance(mb, dict) else type(mb))
if isinstance(mb, dict):
    for k, v in list(mb.items())[:3]:
        print(f"  {k}:", json.dumps(v, indent=2)[:300] if isinstance(v, (dict, list)) else v)
# Check analysis
an = s.get("analysis", {})
print("\nanalysis keys:", list(an.keys())[:10] if isinstance(an, dict) else type(an))
if isinstance(an, dict):
    for k, v in list(an.items())[:5]:
        print(f"  {k}:", json.dumps(v, indent=2)[:300] if isinstance(v, (dict, list)) else v)
# Check idt_baselines
ib = s.get("idt_baselines", {})
print("\nidt_baselines:", json.dumps(ib, indent=2)[:500] if ib else "empty")
