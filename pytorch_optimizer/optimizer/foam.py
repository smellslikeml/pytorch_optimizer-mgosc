import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class FOAM(BaseOptimizer):
    """Factored one-sided Adam-moment SOAP.

    A memory-efficient variant of SOAP for 2D parameters: it rotates the gradient into the eigenbasis of
    the input-side Kronecker factor only, and keeps the Adam second moment in Adafactor-factored form.
    Non-2D parameters fall back to plain Adam.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        shampoo_beta: Decay rate for preconditioner statistics. `None` uses `beta2`.
        weight_decay: Weight decay coefficient.
        precondition_frequency: Number of steps between eigenbasis updates.
        max_precondition_dim: Largest dimension to precondition. Larger dimensions use an identity transform.
        one_sided: Whether to rotate only the input side. `False` also rotates the output side.
        factored_second_moment: Whether to keep the second moment in Adafactor-factored form. `False` keeps
            a dense second moment. Both flags `False` recover SOAP-equivalent behaviour.
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
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FOAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        _, beta2 = group['betas']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(grad)

                if grad.dim() == 2:
                    self.init_foam_state(grad, state, group, beta2)
                else:
                    state['exp_avg_sq'] = torch.zeros_like(grad)

    def init_foam_state(self, grad: torch.Tensor, state, group: ParamGroup, beta2: float) -> None:
        """Initialize the Kronecker factors, eigenbases and second moment for a 2D parameter."""
        state['GG'] = [
            [] if self.use_identity_basis(grad.size(0), group) or group['one_sided'] else self.zero_factor(grad, 0),
            [] if self.use_identity_basis(grad.size(1), group) else self.zero_factor(grad, 1),
        ]
        state['Q'] = None
        state['precondition_frequency'] = group['precondition_frequency']
        state['shampoo_beta'] = group['shampoo_beta'] if group['shampoo_beta'] is not None else beta2

        if group['factored_second_moment']:
            state['exp_avg_sq_row'] = torch.zeros(grad.size(0), device=grad.device, dtype=grad.dtype)
            state['exp_avg_sq_col'] = torch.zeros(grad.size(1), device=grad.device, dtype=grad.dtype)
        else:
            state['exp_avg_sq'] = torch.zeros_like(grad)

        self.update_pre_conditioner(grad, state, step=group['step'])

    def use_identity_basis(self, dim: int, group: ParamGroup) -> bool:
        """Dimensions above `max_precondition_dim` keep an identity basis."""
        return dim > group['max_precondition_dim']

    def zero_factor(self, grad: torch.Tensor, dim: int) -> torch.Tensor:
        """Kronecker factor for `dim` of the gradient: output side for `dim=0`, input side for `dim=1`."""
        size = grad.size(dim)
        return torch.zeros(size, size, device=grad.device, dtype=grad.dtype)

    @staticmethod
    def get_orthogonal_matrix(mat: torch.Tensor) -> torch.Tensor:
        try:
            _, q = torch.linalg.eigh(mat + 1e-30 * torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype))
        except Exception:  # pragma: no cover
            _, q = torch.linalg.eigh(
                mat.to(torch.float64) + 1e-30 * torch.eye(mat.shape[0], device=mat.device, dtype=torch.float64)
            )
            q = q.to(mat.dtype)

        return torch.flip(q, dims=[1])

    def get_orthogonal_matrix_qr(self, state):
        """Compute the eigenbases of the preconditioners using one round of power iteration.

        The refreshed bases re-order coordinates, so the second moment is re-ordered to match.
        """
        matrices = []
        for idx, (m, o) in enumerate(zip(state['GG'], state['Q'])):
            if len(m) == 0:
                matrices.append([])
                continue

            matrix_product = m @ o
            est_eig = (o * matrix_product).sum(dim=0)
            sort_idx = torch.argsort(est_eig, descending=True)

            q, _ = torch.linalg.qr(matrix_product.index_select(1, sort_idx).to(torch.float32))
            matrices.append(q.to(o.dtype))

            if 'exp_avg_sq_row' in state:
                moment = 'exp_avg_sq_row' if idx == 0 else 'exp_avg_sq_col'
                state[moment] = state[moment].index_select(0, sort_idx)
            else:
                dim = idx
                state['exp_avg_sq'] = state['exp_avg_sq'].index_select(dim, sort_idx)

        return matrices

    def project(self, grad: torch.Tensor, state, project_type: str = 'forward') -> torch.Tensor:
        """Rotate `grad` into (`forward`) or out of (`backward`) the Kronecker eigenbases."""
        q_out, q_in = state['Q']

        if len(q_out) > 0:
            grad = q_out.T @ grad if project_type == 'forward' else q_out @ grad

        if len(q_in) > 0:
            grad = grad @ q_in if project_type == 'forward' else grad @ q_in.T

        return grad

    def update_pre_conditioner(self, grad: torch.Tensor, state, step: int) -> None:
        """Update the Kronecker factors and refresh the eigenbases when due."""
        for idx in range(2):
            if len(state['GG'][idx]) > 0:
                outer_product = grad @ grad.T if idx == 0 else grad.T @ grad
                state['GG'][idx].lerp_(outer_product.to(state['GG'][idx].dtype), weight=1.0 - state['shampoo_beta'])

        if state['Q'] is None:
            state['Q'] = [self.get_orthogonal_matrix(m) if len(m) > 0 else [] for m in state['GG']]

        if step > 0 and step % state['precondition_frequency'] == 0:
            state['Q'] = self.get_orthogonal_matrix_qr(state)

    def step_foam(self, p, grad, state, group, beta1: float, beta2: float, step_size: float) -> None:
        """FOAM update for a 2D parameter: eigenbasis rotation with an (optionally factored) second moment."""
        beta2 = state.get('shampoo_beta', beta2)

        grad_projected = self.project(grad, state)

        exp_avg = state['exp_avg']
        exp_avg.lerp_(grad, weight=1.0 - beta1)

        if 'exp_avg_sq_row' in state:
            grad_sq = grad_projected.square()
            state['exp_avg_sq_row'].lerp_(grad_sq.mean(dim=1), weight=1.0 - beta2)
            state['exp_avg_sq_col'].lerp_(grad_sq.mean(dim=0), weight=1.0 - beta2)

            # Adafactor rank-1 reconstruction of the second moment
            row_mean = state['exp_avg_sq_row'].mean().clamp(min=group['eps'])
            de_nom = (
                (state['exp_avg_sq_row'].unsqueeze(1) @ state['exp_avg_sq_col'].unsqueeze(0))
                .div_(row_mean)
                .sqrt_()
                .add_(group['eps'])
            )
        else:
            exp_avg_sq = state['exp_avg_sq']
            exp_avg_sq.lerp_(grad_projected.square(), weight=1.0 - beta2)
            de_nom = exp_avg_sq.sqrt().add_(group['eps'])

        exp_avg_projected = self.project(exp_avg, state)

        p.add_(self.project(exp_avg_projected / de_nom, state, project_type='backward'), alpha=-step_size)

        self.apply_weight_decay(
            p,
            grad=grad,
            lr=group['lr'],
            weight_decay=group['weight_decay'],
            weight_decouple=True,
            fixed_decay=False,
        )

        self.update_pre_conditioner(grad, state, step=group['step'] - 1)

    def step_adam(self, p, grad, state, group, beta1: float, beta2: float, step_size: float) -> None:
        """Plain Adam update for non-2D parameters."""
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

        exp_avg.lerp_(grad, weight=1.0 - beta1)
        exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

        p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-step_size)

        self.apply_weight_decay(
            p,
            grad=grad,
            lr=group['lr'],
            weight_decay=group['weight_decay'],
            weight_decouple=True,
            fixed_decay=False,
        )

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if group['step'] == 1:
                continue

            beta1, beta2 = group['betas']

            step: int = group['step'] - 1
            step_size: float = group['lr']
            step_size *= math.sqrt(self.debias(beta2, step)) / self.debias(beta1, step)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]
                if 'shampoo_beta' in state:
                    self.step_foam(p, grad, state, group, beta1, beta2, step_size)
                else:
                    self.step_adam(p, grad, state, group, beta1, beta2, step_size)

        return loss
