import torch
import torch.nn.functional as F

from utils.run_swd_validation import (
    compute_hf_ratio,
    compute_margin_from_logits,
    compute_patch_consistency_var,
)


def test_hf_ratio_checkerboard_gt_smooth():
    smooth = torch.zeros(1, 1, 32, 32)
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    checker = ((yy + xx) % 2).float().unsqueeze(0).unsqueeze(0)

    smooth_ratio = compute_hf_ratio(smooth, radius_ratio=0.35)
    checker_ratio = compute_hf_ratio(checker, radius_ratio=0.35)

    assert checker_ratio.item() > smooth_ratio.item()


def test_patch_consistency_var_identical_vs_noisy():
    x = torch.randn(2, 4, 16, 16)
    var_same = compute_patch_consistency_var(x, x, patch_size=4)
    assert torch.allclose(var_same, torch.zeros_like(var_same), atol=1e-7)

    y = x + 0.1 * torch.randn_like(x)
    var_noisy = compute_patch_consistency_var(x, y, patch_size=4)
    assert float(var_noisy.mean().item()) > 0.0


def test_margin_from_logits_matches_manual():
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.1, 0.2, 2.0]], dtype=torch.float32)
    target = torch.tensor([0, 1], dtype=torch.long)

    margin, p_t, p_not = compute_margin_from_logits(logits, target)

    probs = F.softmax(logits, dim=1)
    p_t_manual = torch.tensor([probs[0, 0], probs[1, 1]])
    p_not_manual = torch.tensor([torch.max(probs[0, 1:]), torch.max(torch.tensor([probs[1, 0], probs[1, 2]]))])
    margin_manual = p_t_manual - p_not_manual

    assert torch.allclose(p_t, p_t_manual, atol=1e-6)
    assert torch.allclose(p_not, p_not_manual, atol=1e-6)
    assert torch.allclose(margin, margin_manual, atol=1e-6)
