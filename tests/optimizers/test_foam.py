import torch

from pytorch_optimizer import FOAM
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestFoam:
    def test_foam_orthogonal_projection_round_trips(self):
        """The eigenbasis is orthogonal, so projecting forward then backward is the identity."""
        grad = torch.randn(5, 4)
        q = FOAM.get_orthogonal_matrix(grad.t() @ grad)
        assert torch.allclose(q.t() @ q, torch.eye(4), atol=1e-5)

        projected = FOAM.project(grad, q, None)
        restored = FOAM.project(projected, q, None, backward=True)
        assert torch.allclose(restored, grad, atol=1e-5)

    def test_foam_factored_second_moment_is_rank_one_exact(self):
        """For a rank-1 squared gradient the Adafactor row/col reconstruction is exact."""
        row_src = torch.rand(6).abs() + 0.1
        col_src = torch.rand(4).abs() + 0.1
        g2 = torch.outer(row_src, col_src)

        row, col = g2.mean(dim=1), g2.mean(dim=0)
        reconstructed = torch.outer(row, col) / row.mean().clamp_min(1e-12)
        assert torch.allclose(reconstructed, g2, atol=1e-5)

    def test_foam_factored_state_is_cheaper_than_dense(self):
        """Default FOAM keeps one vector per dimension and only the input-side eigenbasis."""
        param = make_parameter((8, 6))
        optimizer = build_optimizer('foam', [param], lr=1e-2, precondition_frequency=1)
        for _ in range(2):
            optimizer.step()

        state = optimizer.state[param]
        assert 'exp_avg_sq_row' in state
        assert 'exp_avg_sq' not in state
        assert tuple(state['exp_avg_sq_row'].shape) == (8,)
        assert tuple(state['exp_avg_sq_col'].shape) == (6,)
        assert state['q_out'] is None  # one-sided: no output-side rotation
        assert state['q_in'] is not None

    def test_foam_dense_two_sided_mode_keeps_dense_moment(self):
        """Turning both options off recovers SOAP's dense moment and two-sided rotation."""
        param = make_parameter((8, 6))
        optimizer = build_optimizer(
            'foam', [param], lr=1e-2, one_sided=False, factored_second_moment=False, precondition_frequency=1
        )
        for _ in range(2):
            optimizer.step()

        state = optimizer.state[param]
        assert 'exp_avg_sq' in state
        assert 'exp_avg_sq_row' not in state
        assert state['q_out'] is not None

    def test_foam_routes_non_2d_params_to_adam(self):
        """Biases and other non-2D parameters get a plain Adam moment, no preconditioner."""
        param = make_parameter((8,))
        optimizer = build_optimizer('foam', [param], lr=1e-2)
        for _ in range(2):
            optimizer.step()

        state = optimizer.state[param]
        assert 'exp_avg_sq' in state
        assert 'q_in' not in state
        assert 'gg_in' not in state
        assert torch.isfinite(param).all()
