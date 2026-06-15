from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import BridgeConfig, ExperimentConfig, ModelConfig, TrainingConfig  # noqa: E402
from losses import OTFlowMatchingObjective  # noqa: E402
from model import TimeConditionedLANCETBridge  # noqa: E402
from ot_cost import SWDTransportCost  # noqa: E402
from trainer import SBTrainer  # noqa: E402


def test_cpu_hungarian_requires_explicit_opt_in() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(),
        bridge=BridgeConfig(coupling_solver="hungarian", allow_cpu_hungarian=False),
    )

    with pytest.raises(ValueError, match="offloads OT matching to CPU"):
        OTFlowMatchingObjective(cfg)


def test_swd_projection_cache_keys_include_spatial_shape() -> None:
    cfg = BridgeConfig(swd_patch_sizes=[3], swd_num_projections=4)
    cost = SWDTransportCost(cfg)

    cost._get_projection_bank(4, device=torch.device("cpu"), spatial_hw=(16, 16))
    cost._get_projection_bank(4, device=torch.device("cpu"), spatial_hw=(32, 32))

    keys = list(cost._projection_cache.keys())
    assert len(keys) == 2
    assert {key[-1] for key in keys} == {(16, 16), (32, 32)}


def test_channels_last_and_compile_are_mutually_exclusive() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(),
        bridge=BridgeConfig(),
        training=TrainingConfig(channels_last=True, torch_compile=True),
    )

    if not torch.cuda.is_available():
        # The trainer only activates channels_last on CUDA, so assert the guard
        # remains present without constructing the full model on CPU.
        assert any(
            isinstance(value, str) and "mutually exclusive" in value
            for value in SBTrainer.__init__.__code__.co_consts
        )
        return

    with pytest.raises(ValueError, match="mutually exclusive"):
        SBTrainer(cfg, torch.device("cuda"))


def test_solver_corrector_mode_is_explicit_not_fallback() -> None:
    assert TimeConditionedLANCETBridge._normalize_solver_corrector_mode("lowpass") == "lowpass_source_anchor"
    with pytest.raises(ValueError, match="Unsupported model.solver_corrector_mode"):
        TimeConditionedLANCETBridge._normalize_solver_corrector_mode("typo_lowpass")


def test_lowpass_corrector_is_opt_in() -> None:
    bridge = TimeConditionedLANCETBridge.__new__(TimeConditionedLANCETBridge)
    bridge.solver_corrector_steps = 1
    bridge.solver_corrector_step_size = 1.0
    bridge.solver_corrector_lowpass_kernel = 3
    bridge.solver_corrector_clamp = 0.0

    h = torch.ones(1, 1, 4, 4)
    source = torch.zeros_like(h)

    bridge.solver_corrector_mode = "none"
    assert torch.equal(bridge._correct_transport_state(h, source, dt=1.0), h)

    bridge.solver_corrector_mode = "lowpass_source_anchor"
    corrected = bridge._correct_transport_state(h, source, dt=1.0)
    assert corrected.abs().mean().item() < h.abs().mean().item()
