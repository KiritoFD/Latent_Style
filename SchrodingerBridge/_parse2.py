"""Parse summary.json - dig into matrix_breakdown and analysis."""
import json

path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010\summary.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== matrix_breakdown ===")
mb = data.get("matrix_breakdown", {})
for k, v in mb.items():
    print(f"  {k}: {type(v).__name__}")
    if isinstance(v, dict):
        for k2, v2 in v.items():
            if isinstance(v2, (int, float)):
                print(f"    {k2}: {v2}")
            elif isinstance(v2, list) and len(v2) > 0:
                print(f"    {k2}: list({len(v2)}) first={v2[0] if len(str(v2[0]))<100 else '...'}")

print()
print("=== analysis ===")
an = data.get("analysis", {})
for k, v in an.items():
    print(f"  {k}: {type(v).__name__}")
    if isinstance(v, dict):
        for k2, v2 in v.items():
            if isinstance(v2, (int, float)):
                print(f"    {k2}: {v2}")
            elif isinstance(v2, dict):
                for k3, v3 in v2.items():
                    if isinstance(v3, (int, float)):
                        print(f"    {k2}.{k3}: {v3}")

print()
print("=== idt_baselines ===")
idt = data.get("idt_baselines", {})
for k, v in idt.items():
    if isinstance(v, (int, float)):
        print(f"  {k}: {v}")

print()
print("=== runtime_observability ===")
ro = data.get("runtime_observability", {})
for k, v in ro.items():
    if isinstance(v, (int, float)):
        print(f"  {k}: {v}")
