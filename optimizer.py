from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")


                # State should be stored in this dictionary
                if p not in self.state:
                    self.state[p] = {}
                    self.state[p]["step"] = 0
                    # Initialize first and second moment vectors
                    self.state[p]["exp_avg"] = torch.zeros_like(p.data)
                    self.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)
                state = self.state[p]
                m,v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Update first and second moments of the gradients
                beta1, beta2 = group["betas"]

                m.mul_(beta1).add_(1 - beta1, grad)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)



                # Bias correction
                if group["correct_bias"]:
                    m = m / (1- beta1**t)
                    v = v / (1-beta2**t)
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                denom = v.sqrt().add_(group["eps"])
                step_size = alpha
                # Update parameters
                p.data.addcdiv_(m, denom, value=-step_size)


                # Add weight decay after the main gradient-based updates.
                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
                # Please note that the learning rate should be incorporated into this update.

        return loss