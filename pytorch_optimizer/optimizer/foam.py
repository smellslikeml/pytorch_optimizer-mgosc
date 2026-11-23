import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class FOAM(BaseOptimizer):
    """Factored One-sided Adam-Moment for Practical and Scalable SOAP.

    A memory-efficient variant of SOAP (eigenvalue-corrected Shampoo). SOAP rotates each 2D weight into the
    eigenbases of *both* of its Kronecker factors and keeps a dense Adam second moment in that rotated space.
    FOAM makes two changes: it rotates only the *input* side (``dim=1`` of a 2D weight) into its eigenbasis, and
    it keeps the second moment in factored Adafactor form (one vector per dimension, reconstructed as their outer
    product) instead of one value per entry. This drops the eigenbasis memory by one factor and the second-moment
    memory from ``m * n`` to ``m + n``. Non-2D parameters (biases, norms, embeddings) fall back to Adam, matching
    the paper's recommended routing.

    Reference: https://arxiv.org/abs/2611.16905

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        shampoo_beta: Decay rate for the input-side preconditioner statistics. `None` uses `beta2`.
        weight_decay: Decoupled weight decay coefficient.
        precondition_frequency: Number of steps between input-side eigenbasis updates.
        max_precondition_dim: Largest input dimension to rotate. A larger input side keeps an identity basis
            (Adafactor-factored Adam without rotation).
        one_sided: Rotate only the input side (FOAM). `False` rotates both sides (recovers SOAP's rotation).
        factored_second_moment: Keep the second moment in factored Adafactor form (FOAM). `False` keeps a dense
            per-entry second moment (recovers SOAP's moment).
        correct_bias: Whether to correct bias in Adam.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-3,
        betas: Betas = (0.95, 0.95),
        shampoo_beta: float | None = None,
        weight_decay: float = 1e-2,
        precondition_frequency: int = 10,
        max_precondition_dim: int = 8192,
        one_sided: bool = True,
        factored_second_moment: bool = True,
        correct_bias: bool = True,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(shampoo_beta, 'shampoo_beta')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(precondition_frequency, 'precondition_frequency')
        self.validate_positive(max_precondition_dim, 'max_precondition_dim')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'shampoo_beta': shampoo_beta,
            'weight_decay': weight_decay,
            'precondition_frequency': precondition_frequency,
            'max_precondition_dim': max_precondition_dim,
            'one_sided': one_sided,
            'factored_second_moment': factored_second_moment,
            'correct_bias': correct_bias,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FOAM'

    @staticmethod
    def get_orthogonal_matrix(mat: torch.Tensor) -> torch.Tensor:
        """Eigenbasis of a symmetric matrix, columns sorted by descending eigenvalue (float32 for stability)."""
        dtype = mat.dtype
        target = mat if dtype in (torch.float32, torch.float64) else mat.to(torch.float32)
        eye = 1e-30 * torch.eye(target.shape[0], device=target.device, dtype=target.dtype)
        try:
            _, q = torch.linalg.eigh(target + eye)
        except Exception:  # pragma: no cover
            _, q = torch.linalg.eigh(target.to(torch.float64) + eye.to(torch.float64))
            q = q.to(torch.float32)
        q = torch.flip(q, dims=[1])
        return q.to(dtype)

    def is_preconditioned(self, p: torch.Tensor, group: ParamGroup) -> bool:
        """FOAM machinery applies to 2D weights whose input side fits; everything else is plain Adam."""
        return p.dim() == 2

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            if p.grad.is_sparse:
                raise NoSparseGradientError(str(self))
            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]
            if len(state) != 0:
                continue

            grad = p.grad
            state['exp_avg'] = torch.zeros_like(grad)

            if not self.is_preconditioned(p, group):  # Adam fallback
                state['exp_avg_sq'] = torch.zeros_like(grad)
                continue

            m, n = grad.shape
            rotate_in: bool = n <= group['max_precondition_dim']
            rotate_out: bool = (not group['one_sided']) and m <= group['max_precondition_dim']

            state['gg_in'] = torch.zeros(n, n, device=grad.device, dtype=grad.dtype) if rotate_in else None
            state['q_in'] = None
            state['gg_out'] = torch.zeros(m, m, device=grad.device, dtype=grad.dtype) if rotate_out else None
            state['q_out'] = None

            if group['factored_second_moment']:
                state['exp_avg_sq_row'] = torch.zeros(m, device=grad.device, dtype=grad.dtype)
                state['exp_avg_sq_col'] = torch.zeros(n, device=grad.device, dtype=grad.dtype)
            else:
                state['exp_avg_sq'] = torch.zeros_like(grad)

    @staticmethod
    def project(grad: torch.Tensor, q_in, q_out, backward: bool = False) -> torch.Tensor:
        """Rotate into (``backward=False``) or out of (``backward=True``) the Kronecker eigenbases.

        Input side rotates ``dim=1`` by ``q_in``; the optional output side rotates ``dim=0`` by ``q_out``. Both
        bases are orthogonal, so the backward projection (transpose) is the exact inverse of the forward one.
        """
        out = grad
        if q_in is not None:
            out = out @ (q_in if not backward else q_in.t())
        if q_out is not None:
            out = (q_out.t() if not backward else q_out) @ out
        return out

    def update_preconditioner(self, grad: torch.Tensor, state, step: int, group: ParamGroup) -> None:
        beta2 = group['betas'][1]
        shampoo_beta = group['shampoo_beta'] if group['shampoo_beta'] is not None else beta2

        if state.get('gg_in') is not None:
            state['gg_in'].lerp_((grad.t() @ grad).to(state['gg_in'].dtype), weight=1.0 - shampoo_beta)
        if state.get('gg_out') is not None:
            state['gg_out'].lerp_((grad @ grad.t()).to(state['gg_out'].dtype), weight=1.0 - shampoo_beta)

        recompute: bool = state.get('q_in') is None or (step > 0 and step % group['precondition_frequency'] == 0)
        if recompute:
            if state.get('gg_in') is not None:
                state['q_in'] = self.get_orthogonal_matrix(state['gg_in'])
            if state.get('gg_out') is not None:
                state['q_out'] = self.get_orthogonal_matrix(state['gg_out'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']
            step: int = group['step']

            step_size: float = group['lr']
            if group['correct_bias']:
                bias_correction1: float = self.debias(beta1, step)
                bias_correction2_sq: float = math.sqrt(self.debias(beta2, step))
                step_size *= bias_correction2_sq / bias_correction1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]
                exp_avg = state['exp_avg']
                exp_avg.lerp_(grad, weight=1.0 - beta1)

                if not self.is_preconditioned(p, group):  # Adam fallback
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)
                    norm_grad = exp_avg / exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    q_in, q_out = state.get('q_in'), state.get('q_out')
                    grad_projected = self.project(grad, q_in, q_out)

                    if group['factored_second_moment']:
                        g2 = grad_projected.square()
                        row, col = state['exp_avg_sq_row'], state['exp_avg_sq_col']
                        row.lerp_(g2.mean(dim=1), weight=1.0 - beta2)
                        col.lerp_(g2.mean(dim=0), weight=1.0 - beta2)
                        # Adafactor rank-1 reconstruction: V[i,j] = row[i] * col[j] / mean(row); exact for rank-1 g2.
                        v = torch.outer(row, col).div_(row.mean().clamp_min(group['eps']))
                        de_nom = v.sqrt_().add_(group['eps'])
                    else:
                        exp_avg_sq = state['exp_avg_sq']
                        exp_avg_sq.lerp_(grad_projected.square(), weight=1.0 - beta2)
                        de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                    exp_avg_projected = self.project(exp_avg, q_in, q_out)
                    norm_grad = self.project(exp_avg_projected / de_nom, q_in, q_out, backward=True)

                    self.update_preconditioner(grad, state, step=step - 1, group=group)

                p.add_(norm_grad, alpha=-step_size)

                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

        return loss
