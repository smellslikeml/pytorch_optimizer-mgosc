import math

import pytest
import torch

from pytorch_optimizer.base.exception import NegativeLRError
from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def run_one_step(lr: float, c: float, gamma: float):
    param = make_parameter(grad=1.0)
    optimizer = build_optimizer('cadam', [param], lr=lr, c=c, gamma=gamma, alpha=1.0)
    optimizer.step()

    return param, optimizer.state[param]['momentum_buffer']


class TestCADAMParameters:
    @pytest.mark.parametrize(
        ('param_name', 'param_value'),
        [
            ('gamma', -0.1),
            ('c', -0.1),
            ('alpha', 0.0),
            ('weight_decay', -1e-3),
            ('eps', -1e-6),
        ],
    )
    def test_invalid_parameters(self, param_name, param_value):
        with pytest.raises(ValueError):
            load_optimizer('cadam')(None, **{param_name: param_value})

    def test_invalid_learning_rate(self):
        with pytest.raises(NegativeLRError):
            load_optimizer('cadam')(None, lr=-1e-2)


def test_reduction_invariant_without_damping():
    """With `c=0` and `gamma=0`, the D and E steps are the identity, leaving the B/C/A-only result."""
    h: float = 0.1
    param, momentum = run_one_step(lr=h, c=0.0, gamma=0.0)

    zeta: float = 1.0 - math.exp(-h)
    torch.testing.assert_close(momentum, torch.full_like(momentum, -h))
    torch.testing.assert_close(param, torch.full_like(param, -h * h / (math.sqrt(zeta) + 1e-8)))


def test_cubic_damping_shrinks_momentum():
    _, damped = run_one_step(lr=0.1, c=1.0, gamma=0.0)
    _, undamped = run_one_step(lr=0.1, c=0.0, gamma=0.0)

    assert damped.abs().sum() < undamped.abs().sum()
