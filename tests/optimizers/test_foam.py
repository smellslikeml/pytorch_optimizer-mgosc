import torch

from pytorch_optimizer.optimizer import load_optimizer
from pytorch_optimizer.optimizer.foam import FOAM
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestFOAM:
    def test_input_side_eigenbasis_round_trip(self):
        param = make_parameter((6, 4), grad=1.0)
        optimizer = build_optimizer('foam', [param], precondition_frequency=1)
        optimizer.step()

        q_in = optimizer.state[param]['Q'][1]
        torch.testing.assert_close(q_in @ q_in.T, torch.eye(4), rtol=1e-4, atol=1e-4)

        state = optimizer.state[param]
        grad = torch.randn(6, 4)
        projected = optimizer.project(grad, state)
        torch.testing.assert_close(optimizer.project(projected, state, project_type='backward'), grad)

    def test_factored_second_moment_reconstruction(self):
        beta2 = 0.95

        # a rank-1 gradient has a rank-1 squared-gradient matrix, which the factored moment reconstructs exactly
        grad = torch.outer(torch.arange(1.0, 7.0), torch.arange(1.0, 5.0))
        param = make_parameter((6, 4), grad=None)
        optimizer = FOAM([param], lr=1e-3, betas=(0.95, beta2), precondition_frequency=10)

        for _ in range(2):
            param.grad = grad.clone()
            optimizer.step()

        state = optimizer.state[param]
        grad_sq = optimizer.project(grad, state).square()
        reconstructed = (state['exp_avg_sq_row'].unsqueeze(1) @ state['exp_avg_sq_col'].unsqueeze(0)) / state[
            'exp_avg_sq_row'
        ].mean()

        torch.testing.assert_close(reconstructed, (1.0 - beta2) * grad_sq)

    def test_default_state_is_factored_and_one_sided(self):
        param = make_parameter((6, 4), grad=1.0)
        optimizer = build_optimizer('foam', [param])
        optimizer.step()

        state = optimizer.state[param]
        assert 'exp_avg_sq' not in state
        assert state['exp_avg_sq_row'].numel() + state['exp_avg_sq_col'].numel() == 10
        assert len(state['GG'][0]) == 0
        assert state['GG'][1].shape == (4, 4)
        assert len(state['Q'][0]) == 0
        assert state['Q'][1].shape == (4, 4)

    def test_non_2d_parameter_uses_adam_fallback(self):
        param = make_parameter((6,), grad=1.0)
        optimizer = build_optimizer('foam', [param], lr=1e-1)

        for _ in range(2):
            optimizer.step()

        state = optimizer.state[param]
        assert set(state) == {'exp_avg', 'exp_avg_sq'}
        assert torch.count_nonzero(param) == 6

    def test_identity_basis_above_max_precondition_dim(self):
        param = make_parameter((6, 4), grad=1.0)
        optimizer = build_optimizer('foam', [param], max_precondition_dim=2, precondition_frequency=1)

        for _ in range(2):
            optimizer.step()

        state = optimizer.state[param]
        assert all(len(q) == 0 for q in state['Q'])
        assert torch.isfinite(param).all()

    def test_factored_two_sided_basis_refresh(self):
        param = make_parameter((6, 4), grad=1.0)
        optimizer = build_optimizer(
            'foam', [param, make_parameter((1,), grad=None)], one_sided=False, precondition_frequency=1
        )

        for _ in range(2):
            optimizer.step()

        state = optimizer.state[param]
        assert state['Q'][0].shape == (6, 6)
        assert torch.isfinite(param).all()

    def test_matches_soap_when_reduction_applies(self):
        torch.manual_seed(42)

        foam_param = make_parameter((6, 4), grad=None)
        soap_param = make_parameter((6, 4), grad=None)
        optimizer = FOAM(
            [foam_param],
            lr=1e-1,
            one_sided=False,
            factored_second_moment=False,
            precondition_frequency=1,
            eps=1e-8,
        )
        soap = load_optimizer('soap')([soap_param], lr=1e-1, precondition_frequency=1)

        for _ in range(3):
            grad = torch.randn(6, 4)
            foam_param.grad = grad.clone()
            soap_param.grad = grad.clone()
            optimizer.step()
            soap.step()

        torch.testing.assert_close(foam_param, soap_param)
