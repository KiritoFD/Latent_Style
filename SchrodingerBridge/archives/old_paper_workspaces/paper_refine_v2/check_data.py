"""Check available artifact data and extract for table update"""
import csv

# Get S2WAT artifact data from summary CSV
path = 'G:/GitHub/Latent_Style/Related_Works/run_511/complete_750/summary_all_tested_metrics_with_ablations.csv'
with open(path) as f:
    for row in csv.DictReader(f):
        m = row['run']
        if m in ('ours_epoch_0007','samst_strict','s2wat_strict'):
            print(f"{m:20s} MUSIQ={row.get('musiq',''):>8s} MANIQA={row.get('maniqa',''):>8s} DISTS={row.get('dists_content',''):>8s} HF-KID={row.get('hf_patch_kid',''):>8s}")
        if m in ('ours_epoch_0007',):
            print(f"  Ours full row keys with data: {[(k,row[k]) for k in row if row[k] and k not in ('run','method','group','path','ablation_label','ablation_purpose')]}")

# Also check summary_artifact_pack_750.csv for more data
path2 = 'G:/GitHub/Latent_Style/Related_Works/run_511/complete_750/summary_artifact_pack_750.csv'
print("\n=== Artifact Pack CSV ===")
try:
    with open(path2) as f:
        for row in csv.DictReader(f):
            print({k: row[k] for k in row if row[k]})
except:
    print("File not found")
