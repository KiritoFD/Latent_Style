"""M3 (redesigned): Per-subband Style Matching Distance

Original M3 measured HH_var but found it identical across all checkpoints.
Root cause: enable_hh_head=False → Flow skips HH; M3 script bug skipped AdaIN.

Redesigned M3 measures: for each checkpoint, run inference WITH style_latent,
DWT decompose output, compute per-subband statistical distance between output
and style_latent. The subband with smallest distance = primary style carrier.

Also measures content→output delta per subband to quantify transport magnitude.
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
OUTPUT_PATH = r"I:\Github\Latent_Style\SchrodingerBridge\mechanism_diagnosis\state\m3_subband_style_distance.json"
NUM_SAMPLES = 50


def load_config(path):
    with open(path) as f:
        raw = json.load(f)
    if "model" in raw and raw["model"].get("solver_family") == "solver_i2sb":
        raw["model"]["solver_family"] = "euler_legacy"
    return ExperimentConfig.from_mapping(raw)


def _subband_stats(x):
    """Return mean and std per subband tensor (B, C, H, W)."""
    return x.float().mean().item(), x.float().std().item()


def measure_subband_distance(model, dataloader, device, num_samples):
    """Run inference WITH style_latent, measure per-subband style matching."""
    model.eval()
    # Accumulate per-subband: output-to-style distance, output-to-content distance
    subbands = ["ll", "lh", "hl", "hh"]
    out_style_dist = {s: [] for s in subbands}  # |mean_out - mean_style| + |std_out - std_style|
    out_content_dist = {s: [] for s in subbands}  # |mean_out - mean_content| + |std_out - std_content|
    style_content_dist = {s: [] for s in subbands}  # baseline: |mean_style - mean_content|
    out_vars = {s: [] for s in subbands}

    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break
            content = batch["content"].to(device).float()
            target_style_id = batch["target_style_id"].to(device)
            # KEY FIX: pass style_latent so AdaIN actually executes
            style_latent = batch.get("target_style", None)
            if style_latent is not None:
                style_latent = style_latent.to(device).float()

            # Run 8-step Euler integration WITH style_latent
            try:
                z_out = model.integrate_transport(
                    content, style_id=target_style_id,
                    num_steps=8, step_size=1.0,
                    style_latent=style_latent,
                )
            except Exception as e:
                print(f"  integrate_transport failed: {e}")
                continue

            # DWT of output, content, style
            o_ll, o_lh, o_hl, o_hh = dwt2_haar(z_out.float())
            c_ll, c_lh, c_hl, c_hh = dwt2_haar(content.float())
            if style_latent is not None:
                s_ll, s_lh, s_hl, s_hh = dwt2_haar(style_latent.float())

            out_subs = {"ll": o_ll, "lh": o_lh, "hl": o_hl, "hh": o_hh}
            content_subs = {"ll": c_ll, "lh": c_lh, "hl": c_hl, "hh": c_hh}
            style_subs = {"ll": s_ll, "lh": s_lh, "hl": s_hl, "hh": s_hh} if style_latent is not None else None

            for s in subbands:
                o_m, o_sd = _subband_stats(out_subs[s])
                c_m, c_sd = _subband_stats(content_subs[s])
                out_vars[s].append(o_sd ** 2)  # variance

                if style_subs is not None:
                    s_m, s_sd = _subband_stats(style_subs[s])
                    # Style matching distance: how close is output to style?
                    out_style_dist[s].append(abs(o_m - s_m) + abs(o_sd - s_sd))
                    # Content drift distance: how far is output from content?
                    out_content_dist[s].append(abs(o_m - c_m) + abs(o_sd - c_sd))
                    # Style-content gap: baseline distance
                    style_content_dist[s].append(abs(s_m - c_m) + abs(s_sd - c_sd))

            count += 1

    results = {}
    for s in subbands:
        results[s] = {
            "out_var": sum(out_vars[s]) / max(len(out_vars[s]), 1),
            "out_to_style_dist": sum(out_style_dist[s]) / max(len(out_style_dist[s]), 1) if out_style_dist[s] else None,
            "out_to_content_dist": sum(out_content_dist[s]) / max(len(out_content_dist[s]), 1) if out_content_dist[s] else None,
            "style_to_content_dist": sum(style_content_dist[s]) / max(len(style_content_dist[s]), 1) if style_content_dist[s] else None,
        }
        # Style transfer ratio: how much of the style-content gap was closed?
        if results[s]["out_to_style_dist"] is not None and results[s]["style_to_content_dist"] is not None:
            gap = results[s]["style_to_content_dist"]
            if gap > 1e-8:
                # transfer_ratio = 1 - (out_to_style / style_to_content)
                # = fraction of style gap closed by transport
                results[s]["style_transfer_ratio"] = max(0.0, 1.0 - results[s]["out_to_style_dist"] / gap)
            else:
                results[s]["style_transfer_ratio"] = 0.0
        else:
            results[s]["style_transfer_ratio"] = None

    results["num_samples"] = count
    return results


def main():
    print("Loading config...")
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda")

    data_cfg = config.data
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

            stats = measure_subband_distance(model, dataloader, device, NUM_SAMPLES)
            stats["epoch"] = epoch
            stats["checkpoint_dir"] = os.path.basename(ckpt_dir)
            results.append(stats)

            print(f"  Subband style_transfer_ratio:")
            for s in ["ll", "lh", "hl", "hh"]:
                r = stats[s]["style_transfer_ratio"]
                v = stats[s]["out_var"]
                print(f"    {s}: transfer_ratio={r:.4f}  out_var={v:.6f}")

    # Identity baseline (no transfer)
    print("\n=== Identity baseline ===")
    id_vars = {"ll": [], "lh": [], "hl": [], "hh": []}
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            if count >= NUM_SAMPLES:
                break
            content = batch["content"].to(device).float()
            ll, lh, hl, hh = dwt2_haar(content)
            id_vars["ll"].append(ll.float().var().item())
            id_vars["lh"].append(lh.float().var().item())
            id_vars["hl"].append(hl.float().var().item())
            id_vars["hh"].append(hh.float().var().item())
            count += 1
    identity_stats = {
        "epoch": 0,
        "checkpoint_dir": "identity",
        "num_samples": count,
    }
    for s in ["ll", "lh", "hl", "hh"]:
        identity_stats[s] = {
            "out_var": sum(id_vars[s]) / len(id_vars[s]),
            "out_to_style_dist": None,
            "out_to_content_dist": 0.0,
            "style_to_content_dist": None,
            "style_transfer_ratio": 0.0,
        }
    results.insert(0, identity_stats)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    print("\n=== SUMMARY: Style Transfer Ratio by Subband ===")
    print(f"{'Checkpoint':<35} {'LL':>8} {'LH':>8} {'HL':>8} {'HH':>8}")
    for r in results:
        name = f"{r['checkpoint_dir']}/ep{r['epoch']:04d}"
        ll_r = r["ll"]["style_transfer_ratio"] or 0
        lh_r = r["lh"]["style_transfer_ratio"] or 0
        hl_r = r["hl"]["style_transfer_ratio"] or 0
        hh_r = r["hh"]["style_transfer_ratio"] or 0
        print(f"  {name:<33} {ll_r:>8.4f} {lh_r:>8.4f} {hl_r:>8.4f} {hh_r:>8.4f}")


if __name__ == "__main__":
    main()
