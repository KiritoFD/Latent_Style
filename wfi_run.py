import sys, json
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/mnt/i/Github/Latent_Style/SchrodingerBridge/tools")
from probe_620_fog_whiteness_index import evaluate_directory

dirs = {
    "intrinsic_e8": "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/full_eval/epoch_0008/images",
    "intrinsic_e1": "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/full_eval/epoch_0001/images",
    "film_e3": "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_film_formal/full_eval/epoch_0003/images",
}
for name, path in dirs.items():
    r = evaluate_directory(path, sample_count=20)
    m = r.get("metrics", {})
    wfi = m.get("avg_wfi_score", "?")
    cr = m.get("avg_contrast_ratio", "?")
    dr = m.get("avg_dynamic_range", "?")
    sat = m.get("avg_saturation_mean", "?")
    ls = m.get("avg_luminance_std", "?")
    print(f"{name}: wfi={wfi} cr={cr} dr={dr} sat={sat} lum_std={ls}")
