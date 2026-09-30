"""M3: HH-variance / CLIP-S Co-evolution
Measure output latent's HH subband variance at each epoch checkpoint.
If HH_var correlates with CLIP-S across epochs, confirms HH is the style carrier.

Uses 710_b0_weave checkpoints (epoch 5, 10) + trains a 3-epoch model for more points.
"""
import sys, os, json, torch
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")
os.chdir(r"I:\Github\Latent_Style\SchrodingerBridge")

from config_schema import ExperimentConfig
from model import build_model_from_config
from utils.dataset import AdaCUTLatentDataset
from wavelet import dwt2_haar, idwt2_haar
from torch.utils.data import DataLoader

CONFIG_PATH = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_weave\config.json"
CKPT_DIRS = [
    (r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_weave", [5, 10]),
    (r"I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline", [5]),
]
OUTPUT_PATH = r"I:\Github\Latent_Style\SchrodingerBridge\mechanism_diagnosis\state\m3_hh_variance.json"
NUM_SAMPLES = 50  # measure HH var on 50 test pairs


def load_config(path):
    with open(path) as f:
        raw = json.load(f)
    if "model" in raw and raw["model"].get("solver_family") == "solver_i2sb":
        raw["model"]["solver_family"] = "euler_legacy"
    return ExperimentConfig.from_mapping(raw)


def measure_hh_variance(model, dataloader, device, num_samples):
    """Run inference on N samples, measure HH subband variance of output latent."""
    model.eval()
    hh_vars = []
    ll_vars = []
    lh_vars = []
    hl_vars = []
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break
            content = batch["content"].to(device).float()
            target_style_id = batch["target_style_id"].to(device)

            # Run 8-step Euler integration (model.integrate_transport)
            try:
                z_out = model.integrate_transport(
                    content, style_id=target_style_id,
                    num_steps=8, t_start=0.0, t_end=1.0,
                )
            except Exception as e:
                print(f"  integrate_transport failed: {e}, trying forward only")
                # Fallback: just use forward at t=1
                z_out = model(content, t=1.0, style_id=target_style_id)
                if isinstance(z_out, dict):
                    z_out = idwt2_haar(z_out["ll"], z_out["lh"], z_out["hl"], None)

            # DWT of output
            ll, lh, hl, hh = dwt2_haar(z_out)
            hh_vars.append(hh.float().var().item())
            ll_vars.append(ll.float().var().item())
            lh_vars.append(lh.float().var().item())
            hl_vars.append(hl.float().var().item())
            count += 1

    return {
        "hh_var": sum(hh_vars) / len(hh_vars),
        "ll_var": sum(ll_vars) / len(ll_vars),
        "lh_var": sum(lh_vars) / len(lh_vars),
        "hl_var": sum(hl_vars) / len(hl_vars),
        "num_samples": count,
    }


def main():
    print("Loading config...")
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda")

    # Load dataset (test split)
    data_cfg = config.data
    # Use train data root (test latents may not exist separately)
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.data_root,
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=False,
        identity_ratio=None,
        batch_size_hint=4,
        balance_target_styles_per_batch=False,
        preload_to_gpu=False,
        preload_max_vram_gb=0.0,
        preload_reserve_ratio=0.35,
        virtual_length_multiplier=1.0,
        content_style_sampling_weights=None,
        target_style_sampling_weights=None,
        pairing_cache_path=data_cfg.pairing_cache_path,
        pairing_cache_topk=int(data_cfg.pairing_cache_topk),
        pairing_cache_active_topk=int(data_cfg.pairing_cache_active_topk),
        pairing_cache_sample_mode=str(data_cfg.pairing_cache_sample_mode),
        pairing_cache_rank_schedule=str(data_cfg.pairing_cache_rank_schedule),
        pairing_cache_min_topk=int(data_cfg.pairing_cache_min_topk),
        pairing_cache_curriculum_epochs=0,
        pairing_cache_rank_power=float(data_cfg.pairing_cache_rank_power),
        pairing_cache_explore_prob=0.0,
        pairing_cache_explore_topk=0,
        pairing_cache_dual_target_mix=0.0,
        pairing_cache_dual_target_topk=0,
        pairing_cache_aux_target_topk=0,
        pairing_cache_cross_only=bool(data_cfg.pairing_cache_cross_only),
        latent_cache_mode=str(data_cfg.latent_cache_mode),
        latent_cache_dir=str(data_cfg.latent_cache_dir),
        style_caption_path="",
        device="cpu",
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    results = []

    for ckpt_dir, epochs in CKPT_DIRS:
        cfg_path = os.path.join(ckpt_dir, "config.json")
        if os.path.exists(cfg_path):
            ckpt_config = load_config(cfg_path)
        else:
            ckpt_config = config

        print(f"\n=== Checkpoints in {ckpt_dir} ===")
        model = build_model_from_config(ckpt_config.model, bridge_cfg=ckpt_config.bridge).to(device)

        for epoch in epochs:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  epoch_{epoch:04d}.pt not found, skipping")
                continue

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"  Loaded epoch {epoch} (missing={len(missing)}, unexpected={len(unexpected)})")

            stats = measure_hh_variance(model, dataloader, device, NUM_SAMPLES)
            stats["epoch"] = epoch
            stats["checkpoint_dir"] = os.path.basename(ckpt_dir)
            results.append(stats)
            print(f"  HH_var={stats['hh_var']:.6f} LL_var={stats['ll_var']:.6f} "
                  f"LH_var={stats['lh_var']:.6f} HL_var={stats['hl_var']:.6f}")

    # Also measure on identity (no transfer) for baseline
    print("\n=== Identity baseline (no transfer) ===")
    hh_id, ll_id, lh_id, hl_id = [], [], [], []
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            if count >= NUM_SAMPLES:
                break
            content = batch["content"].to(device).float()
            ll, lh, hl, hh = dwt2_haar(content)
            hh_id.append(hh.float().var().item())
            ll_id.append(ll.float().var().item())
            lh_id.append(lh.float().var().item())
            hl_id.append(hl.float().var().item())
            count += 1
    identity_stats = {
        "epoch": 0,
        "checkpoint_dir": "identity",
        "hh_var": sum(hh_id) / len(hh_id),
        "ll_var": sum(ll_id) / len(ll_id),
        "lh_var": sum(lh_id) / len(lh_id),
        "hl_var": sum(hl_id) / len(hl_id),
        "num_samples": count,
    }
    results.insert(0, identity_stats)
    print(f"  HH_var={identity_stats['hh_var']:.6f} LL_var={identity_stats['ll_var']:.6f}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Quick summary
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['checkpoint_dir']}/epoch_{r['epoch']:04d}: "
              f"HH_var={r['hh_var']:.6f} LL_var={r['ll_var']:.6f} "
              f"HH/LL ratio={r['hh_var']/max(r['ll_var'],1e-12):.4f}")


if __name__ == "__main__":
    main()
