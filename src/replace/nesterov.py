"""Nesterov's accelerated gradient method as used by ePlace / RePlAce.

The step length is predicted from a local Lipschitz-constant estimate,
    alpha = ||v_k - v_{k-1}|| / ||grad(v_k) - grad(v_{k-1})||,
and backtracking re-tries a step while the new estimate is clearly smaller
than the one used, i.e. the step was too long.
"""

import math
from typing import Callable

import torch

GradFn = Callable[[torch.Tensor], tuple[torch.Tensor, dict]]


def _lipschitz_step(v0: torch.Tensor, g0: torch.Tensor, v1: torch.Tensor, g1: torch.Tensor) -> float:
    return ((v1 - v0).norm() / (g1 - g0).norm().clamp(min=1e-30)).item()


class Nesterov:
    """Minimizes a function given its (preconditioned) gradient.

    `grad_fn(v)` returns (gradient at v, stats); stats of the latest accepted
    reference solution are exposed as `self.stats`. `project(x)` keeps
    solutions inside the feasible region.
    """

    def __init__(self, x0: torch.Tensor, grad_fn: GradFn, project: Callable, initial_move: float, max_backtracks: int = 10):
        self.grad_fn, self.project, self.max_backtracks = grad_fn, project, max_backtracks
        self.u = x0.clone()  # major solution
        self.v = x0.clone()  # reference solution (where gradients are taken)
        self.a = 1.0  # Nesterov momentum parameter
        self.g, self.stats = grad_fn(self.v)
        # Initial step length: Lipschitz estimate from a small probe step
        # moving the fastest object by `initial_move`.
        probe = project(self.v - initial_move / self.g.abs().max().clamp(min=1e-30) * self.g)
        g_probe, _ = grad_fn(probe)
        self.step_length = _lipschitz_step(self.v, self.g, probe, g_probe)

    def step(self) -> None:
        a_next = (1.0 + math.sqrt(4.0 * self.a**2 + 1.0)) / 2.0
        for _ in range(self.max_backtracks):
            u_next = self.project(self.v - self.step_length * self.g)
            v_next = self.project(u_next + (self.a - 1.0) / a_next * (u_next - self.u))
            g_next, stats = self.grad_fn(v_next)
            new_step = _lipschitz_step(self.v, self.g, v_next, g_next)
            accept = new_step > 0.95 * self.step_length
            self.step_length = new_step
            if accept:
                break
        self.u, self.v, self.g, self.a, self.stats = u_next, v_next, g_next, a_next, stats
