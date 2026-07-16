import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from internal_dynamics import InternalDynamicsState


def test_internal_transition_requires_gate_reversal_and_relative_gradient_drop():
    state = InternalDynamicsState()
    rows = [
        (1, 0.173068, 1.4727),
        (2, 0.172264, 1.4328),
        (3, 0.172009, 1.0868),
        (4, 0.175722, 0.1634),
        (5, 0.179492, 0.1510),
    ]
    transitions = []
    for epoch, gate, ratio in rows:
        metrics = {
            "internal_probe_gate_mean": gate,
            "internal_probe_shared_ll_hf_grad_ratio": ratio,
        }
        transitions.append(
            state.update(
                epoch,
                metrics,
                min_epoch=3,
                gate_delta_threshold=0.0,
                shared_ratio_drop_threshold=0.65,
            )
        )
    assert transitions == [False, False, False, True, False]
    assert state.transition_epoch == 4


def test_internal_transition_rejects_ratio_crossing_while_gate_is_contracting():
    state = InternalDynamicsState(previous_gate_mean=0.18, previous_shared_ll_hf_ratio=1.2)
    metrics = {
        "internal_probe_gate_mean": 0.17,
        "internal_probe_shared_ll_hf_grad_ratio": 0.8,
    }
    assert not state.update(
        4,
        metrics,
        min_epoch=3,
        gate_delta_threshold=0.0,
        shared_ratio_drop_threshold=0.65,
    )


def test_relative_drop_is_invariant_to_absolute_gradient_scale():
    state = InternalDynamicsState()
    rows = [
        (1, 0.1731, 0.87),
        (2, 0.1723, 0.67),
        (3, 0.1720, 0.42),
        (4, 0.1757, 0.24),
    ]
    transitions = []
    for epoch, gate, ratio in rows:
        metrics = {
            "internal_probe_gate_mean": gate,
            "internal_probe_shared_ll_hf_grad_ratio": ratio,
        }
        transitions.append(state.update(
            epoch,
            metrics,
            min_epoch=3,
            gate_delta_threshold=0.0,
            shared_ratio_drop_threshold=0.65,
        ))
    assert transitions == [False, False, False, True]
