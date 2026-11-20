import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class MGOSC(BaseOptimizer):
    """Moment-Guided Optimizer with Sign-Consistency.

    Adam-style update scaled per-coordinate by a sign-consistency reliability gate. Alongside the gradient
    moments, a sign EMA ``s`` tracks whether a coordinate's gradient keeps the same direction; the gate
    ``r = |s|^gamma`` (optionally bias-corrected, optionally floored at ``r_min``) shrinks the step on
    coordinates whose sign flips often, which the paper reports improves robustness under label noise.
    With ``gamma = 0`` the gate is ``1`` everywhere and the update reduces to Adam.

    Reference: Moment-Guided Optimizers for Noisy Deep Learning (https://arxiv.org/abs/2610.37712).

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the gradient mean and squared gradients.
        beta_s: Decay rate for the sign-consistency EMA.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        gamma: Exponent of the reliability gate. ``0`` recovers Adam.
        r_min: Lower bound on the gate when ``> 0``; keeps a minimum step on inconsistent coordinates.
        bias_correction_s: Bias-correct the sign EMA (useful in early steps).
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        beta_s: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        gamma: float = 1.0,
        r_min: float = 0.0,
        bias_correction_s: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(beta_s, 'beta_s', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(gamma, 'gamma')
        self.validate_range(r_min, 'r_min', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'beta_s': beta_s,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'gamma': gamma,
            'r_min': r_min,
            'bias_correction_s': bias_correction_s,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'MGOSC'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['sign_ema'] = torch.zeros_like(p)

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
            beta_s: float = group['beta_s']
            gamma: float = group['gamma']
            r_min: float = group['r_min']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = self.debias(beta2, group['step']) ** 0.5
            bias_correction_s: float = self.debias(beta_s, group['step']) if group['bias_correction_s'] else 1.0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq, sign_ema = state['exp_avg'], state['exp_avg_sq'], state['sign_ema']

                p, grad, exp_avg, exp_avg_sq, sign_ema = self.view_as_real(
                    p, grad, exp_avg, exp_avg_sq, sign_ema
                )

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg.lerp_(grad, weight=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                sign_ema.lerp_(torch.sign(grad), weight=1.0 - beta_s)

                de_nom = (exp_avg_sq / bias_correction2_sq ** 2).sqrt_().add_(group['eps'])
                update = (exp_avg / bias_correction1) / de_nom

                if gamma > 0.0:
                    gate = (sign_ema / bias_correction_s).abs_().pow_(gamma)
                    if r_min > 0.0:
                        gate.clamp_(min=r_min, max=1.0)
                    update.mul_(gate)

                p.add_(update, alpha=-group['lr'])

        return loss
