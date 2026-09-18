"""Euclidean projection onto a capped simplex for diagnostic gate budgets."""

from __future__ import annotations

import torch


@torch.no_grad()
def project_capped_simplex(
    values: torch.Tensor,
    *,
    target_sum: float,
    iterations: int = 80,
    sum_tolerance: float = 1e-4,
) -> torch.Tensor:
    """Project ``values`` onto ``0 <= x <= 1`` and ``sum(x)=target_sum``.

    The Euclidean projection has the form ``clamp(values - tau, 0, 1)``.
    Bisection finds the scalar ``tau``.  Computation is performed in FP64 and
    converted back to the input floating dtype at the end.
    """

    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be a non-empty rank-1 tensor")
    if not values.is_floating_point():
        raise TypeError("values must use a floating dtype")
    if not torch.isfinite(values).all():
        raise ValueError("values contain non-finite entries")
    if not 0.0 <= float(target_sum) <= values.numel():
        raise ValueError("target_sum must be in [0, number of entries]")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    work = values.detach().to(dtype=torch.float64)
    target = torch.tensor(float(target_sum), dtype=torch.float64, device=work.device)
    if target_sum == 0:
        return torch.zeros_like(values)
    if target_sum == values.numel():
        return torch.ones_like(values)

    lower = work.min() - 1.0
    upper = work.max()
    for _ in range(iterations):
        tau = (lower + upper) / 2.0
        projected_sum = (work - tau).clamp(0.0, 1.0).sum()
        if projected_sum > target:
            lower = tau
        else:
            upper = tau
    projected = (work - (lower + upper) / 2.0).clamp(0.0, 1.0)
    result = projected.to(dtype=values.dtype)
    error = abs(float(result.double().sum().item()) - float(target_sum))
    if error > sum_tolerance:
        raise AssertionError(
            f"capped-simplex projection sum error {error} exceeds {sum_tolerance}"
        )
    if not bool(((result >= 0) & (result <= 1)).all()):
        raise AssertionError("capped-simplex projection violated [0,1] bounds")
    return result
