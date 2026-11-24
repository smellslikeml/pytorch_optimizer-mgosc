import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT


class CADAM(BaseOptimizer):
    """Cubically-damped Adam via operator splitting of a momentum ODE with nonlinear damping.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Step size of the split integrator.
        gamma: Linear friction coefficient of the momentum decay.
        c: Cubic damping coefficient of the momentum decay.
        alpha: Relaxation rate of the second-moment Ornstein-Uhlenbeck update.
        weight_decay: Weight decay coefficient.
        eps: Term added to the denominator to improve numerical stability.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        gamma: float = 1.0,
        c: float = 1.0,
        alpha: float = 1.0,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(gamma, 'gamma')
        self.validate_non_negative(c, 'c')
        self.validate_positive(alpha, 'alpha')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: Defaults = {
            'lr': lr,
            'gamma': gamma,
            'c': c,
            'alpha': alpha,
            'weight_decay': weight_decay,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'CADAM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

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
                state['momentum_buffer'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            lr: float = group['lr']
            gamma: float = group['gamma']
            c: float = group['c']
            alpha: float = group['alpha']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                exp_avg, exp_avg_sq = state['momentum_buffer'], state['exp_avg_sq']

                self.apply_weight_decay(
                    p=p,
                    grad=None,
                    lr=lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

                # B step: gradient force on the momentum.
                exp_avg.add_(grad, alpha=-lr)

                # C step: second-moment Ornstein-Uhlenbeck update.
                exp_avg_sq.mul_(math.exp(-alpha * lr)).addcmul_(
                    grad, grad, value=(1.0 - math.exp(-alpha * lr)) / alpha
                )

                # A step: Adam-normalized position update.
                de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                p.addcdiv_(exp_avg, de_nom, value=lr)

                # D step: cubic (nonlinear) damping of the momentum.
                exp_avg.div_(torch.sqrt(2 * c * lr * exp_avg.pow(2) + 1.0))

                # E step: linear friction of the momentum.
                exp_avg.mul_(math.exp(-gamma * lr))

        return loss
