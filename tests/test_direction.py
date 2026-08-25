import torch

from argos.config import ArgosConfig
from argos.direction import compute_refusal_directions, compute_refusal_directions_indexed


def _config():
    return ArgosConfig(model_id="dummy", selected_layers=["resid_pre"])


def test_zero_norm_layer_is_excluded_not_nan():
    hidden = 8
    harmful_acts = {
        "resid_pre": {
            0: torch.ones(4, hidden),
            1: torch.tensor([[1.0] * hidden, [2.0] * hidden, [3.0] * hidden, [4.0] * hidden]),
        }
    }
    harmless_acts = {
        "resid_pre": {
            0: torch.ones(4, hidden),
            1: torch.zeros(4, hidden),
        }
    }

    directions = compute_refusal_directions(harmful_acts, harmless_acts, _config())

    assert len(directions) == 1
    for d in directions:
        assert not torch.isnan(d).any()
        assert abs(d.norm().item() - 1.0) < 1e-5


def test_indexed_directions_skip_zero_norm_layer_and_keep_layer_id():
    hidden = 8
    harmful_acts = {
        "resid_pre": {
            0: torch.ones(4, hidden),
            1: torch.tensor([[1.0] * hidden, [2.0] * hidden, [3.0] * hidden, [4.0] * hidden]),
        }
    }
    harmless_acts = {
        "resid_pre": {
            0: torch.ones(4, hidden),
            1: torch.zeros(4, hidden),
        }
    }

    indexed = compute_refusal_directions_indexed(harmful_acts, harmless_acts, _config())

    assert len(indexed) == 1
    act_name, layer_idx, direction = indexed[0]
    assert act_name == "resid_pre"
    assert layer_idx == 1
    assert not torch.isnan(direction).any()
