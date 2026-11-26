"""Small helpers for choosing two-mode momentum constants.

age(beta)          mean age (in steps) of an EMA with decay beta: beta / (1 - beta)
beta_for_age(A)    inverse of age()
kernel_age(bf, bs, w)            mean age of w * EMA_bf + (1 - w) * EMA_bs
fast_weight_for_age(bf, bs, A)   the w that gives mean age A for units bf, bs (A must lie between their ages)
switch_step(total_steps, decay_start_fraction)   the step at which to switch the two relaxation modes on
"""
from __future__ import annotations


def age(beta: float) -> float:
    assert 0.0 <= beta < 1.0
    return beta / (1.0 - beta)


def beta_for_age(a: float) -> float:
    assert a >= 0.0
    return a / (1.0 + a)


def kernel_age(beta_fast: float, beta_slow: float, w_fast: float) -> float:
    return w_fast * age(beta_fast) + (1.0 - w_fast) * age(beta_slow)


def fast_weight_for_age(beta_fast: float, beta_slow: float, mean_age: float) -> float:
    af, as_ = age(beta_fast), age(beta_slow)
    assert min(af, as_) <= mean_age <= max(af, as_), (af, as_, mean_age)
    return (as_ - mean_age) / (as_ - af)


def switch_step(total_steps: int, decay_start_fraction: float) -> int:
    """Switch the two relaxation modes on where the learning-rate decay begins (e.g. 0.3 for a 70% linear cooldown)."""
    assert 0.0 <= decay_start_fraction < 1.0
    return int(round(total_steps * decay_start_fraction))
