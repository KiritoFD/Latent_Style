"""M1: Gradient Spectrum Analysis
Prove low-frequency gradients dominate training — causal evidence for frequency dominance.

Measures:
  (A) Input-gradient spectrum: DWT(grad_x_t) energy shares across LL/LH/HL/HH.
  (B) Per-head parameter gradient norms: ||grad(head_ll)|| vs ||grad(head_lh)|| vs ||grad(head_hl)||.
  (C) Per-subband loss values: loss_ll vs loss_lh vs loss_hl (absolute, unweighted).
"""
import sys, os, json, torch
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")
os.chdir(r"I:\Github\Latent_Style\SchrodingerBridge")

from config_schema import ExperimentConfig
from model import build_model_from_config
from utils.dataset import AdaCUTLatentDataset
from wavelet import dwt2_haar
from torch.utils.data import DataLoader


def load_config_bypass_validation(path):
    """Load config bypassing i2sb validation (incompatible with flow_matching)."""
    with open(path) as f:
        raw = json.load(f)
    # Patch solver_family to avoid validation error
    if "model" in raw and raw["model"].get("solver_family") == "solver_i2sb":
        raw["model"]["solver_family"] = "euler_legacy"
    return ExperimentConfig.from_mapping(raw)

CONFIG_PATH = r"I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline\config.json"
CKPT_PATH = r"I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline\epoch_0005.pt"
OUTPUT_PATH = r"I:\Github\Latent_Style\SchrodingerBridge\mechanism_diagnosis\state\m1_gradient_spectrum.json"
NUM_BATCHES = 5
T_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
BATCH_SIZE = 8


def head_grad_norm(head):
    """Total L2 norm of all parameter gradients in a head."""
    total = 0.0
    count = 0
    for p in head.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
            count += p.numel()
    return total ** 0.5


def analyze_one_batch(model, content, target, target_style_id, t_val, w_ll, w_lh, w_hl):
    """Analyze gradient spectrum for one (content, target, t) triple."""
    device = content.device
    t = torch.full((content.shape[0],), t_val, device=device, dtype=content.dtype)
    x_t = (1.0 - t_val) * content + t_val * target
    x_t.requires_grad_(True)

    # Forward
    v_dict = model(x_t, t=t, style_id=target_style_id, style_latent=target)

    # Per-subband targets
    target_delta = target - content
    t_ll, t_lh, t_hl, t_hh = dwt2_haar(target_delta)

    # Losses (unweighted)
    loss_ll = ((v_dict["ll"].float() - t_ll.float()) ** 2).mean()
    loss_lh = ((v_dict["lh"].float() - t_lh.float()) ** 2).mean()
    loss_hl = ((v_dict["hl"].float() - t_hl.float()) ** 2).mean()

    # Weighted total (matches training)
    loss = w_ll * loss_ll + w_lh * loss_lh + w_hl * loss_hl

    # Backward
    model.zero_grad()
    loss.backward()

    # (A) Input-gradient spectrum
    grad_x = x_t.grad.detach()
    g_ll, g_lh, g_hl, g_hh = dwt2_haar(grad_x)
    e_ll = (g_ll.float() ** 2).sum().item()
    e_lh = (g_lh.float() ** 2).sum().item()
    e_hl = (g_hl.float() ** 2).sum().item()
    e_hh = (g_hh.float() ** 2).sum().item()
    total_e = max(e_ll + e_lh + e_hl + e_hh, 1e-12)

    # (B) Head parameter gradient norms
    hg_ll = head_grad_norm(model.head_ll)
    hg_lh = head_grad_norm(model.head_lh)
    hg_hl = head_grad_norm(model.head_hl)

    return {
        "t": t_val,
        "loss_ll": loss_ll.item(),
        "loss_lh": loss_lh.item(),
        "loss_hl": loss_hl.item(),
        "loss_total": loss.item(),
        "grad_energy_ll": e_ll,
        "grad_energy_lh": e_lh,
        "grad_energy_hl": e_hl,
        "grad_energy_hh": e_hh,
        "grad_share_ll": e_ll / total_e,
        "grad_share_lh": e_lh / total_e,
        "grad_share_hl": e_hl / total_e,
        "grad_share_hh": e_hh / total_e,
        "head_grad_norm_ll": hg_ll,
        "head_grad_norm_lh": hg_lh,
        "head_grad_norm_hl": hg_hl,
    }


def main():
    print("Loading config...")
    config = load_config_bypass_validation(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    print("Building model...")
    model = build_model_from_config(config.model, bridge_cfg=config.bridge).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"Unexpected keys (ignored): {unexpected[:5]}...")
    model.eval()  # disable dropout, but gradients still flow
    print(f"Model loaded from {CKPT_PATH}")

    # Load data
    print("Loading dataset...")
    data_cfg = config.data
    train_cfg = config.training
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.data_root,
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=False,  # deterministic for analysis
        identity_ratio=data_cfg.identity_ratio,
        batch_size_hint=BATCH_SIZE,
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
        device="cpu",  # load to CPU, move batch to GPU manually
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Loss weights
    w_ll = float(getattr(config.bridge, "spectral_w_ll", 0.3))
    w_lh = float(getattr(config.bridge, "spectral_w_lh", 1.0))
    w_hl = float(getattr(config.bridge, "spectral_w_hl", 1.0))
    print(f"Loss weights: w_ll={w_ll}, w_lh={w_lh}, w_hl={w_hl}")

    # Analyze
    all_results = []
    batch_iter = iter(dataloader)
    for batch_idx in range(NUM_BATCHES):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        content = batch["content"].to(device).float()
        target = batch["target_style"].to(device).float()
        target_style_id = batch["target_style_id"].to(device)
        print(f"\nBatch {batch_idx}: content {content.shape}, target {target.shape}")

        for t_val in T_VALUES:
            r = analyze_one_batch(model, content, target, target_style_id, t_val, w_ll, w_lh, w_hl)
            r["batch"] = batch_idx
            all_results.append(r)
            print(f"  t={t_val:.1f}: LL_grad={r['grad_share_ll']:.3f} LH={r['grad_share_lh']:.3f} "
                  f"HL={r['grad_share_hl']:.3f} HH={r['grad_share_hh']:.3f} | "
                  f"loss_ll={r['loss_ll']:.4f} loss_lh={r['loss_lh']:.4f} loss_hl={r['loss_hl']:.4f}")

    # Aggregate
    print("\n=== AGGREGATED (mean across all batches and t values) ===")
    agg = {}
    for key in ["grad_share_ll", "grad_share_lh", "grad_share_hl", "grad_share_hh",
                 "loss_ll", "loss_lh", "loss_hl",
                 "head_grad_norm_ll", "head_grad_norm_lh", "head_grad_norm_hl"]:
        vals = [r[key] for r in all_results]
        agg[key + "_mean"] = sum(vals) / len(vals)
        agg[key + "_std"] = (sum((v - agg[key + "_mean"]) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {key}: {agg[key + '_mean']:.4f} ± {agg[key + '_std']:.4f}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output = {"per_sample": all_results, "aggregated": agg,
              "config": {"w_ll": w_ll, "w_lh": w_lh, "w_hl": w_hl,
                         "num_batches": NUM_BATCHES, "t_values": T_VALUES,
                         "checkpoint": "epoch_0005"}}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
