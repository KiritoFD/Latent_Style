"""读取 SaMam 评估结果, 查证 0.7222 是 transfer 还是 all_pairs."""
import json, os

paths = [
    r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_reeval\samam_latent_step1000_reeval\summary.json",
    r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\samam\summary.json",
]

for path in paths:
    print(f"=== {path} ===")
    if not os.path.exists(path):
        print("  NOT FOUND")
        continue
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # 打印 analysis 部分
    analysis = d.get("analysis", {})
    for section_name in ["all_pairs_overview", "style_transfer_ability", "identity_reconstruction"]:
        section = analysis.get(section_name, {})
        print(f"\n  --- {section_name} ---")
        for k, v in section.items():
            if isinstance(v, (int, float)):
                print(f"    {k}: {v}")

    # 打印 checkpoint
    print(f"\n  checkpoint: {d.get('checkpoint', 'N/A')}")
    print(f"  timestamp: {d.get('timestamp', 'N/A')}")
    print()
