"""Recommended settings. Pass one of them to ``PhysMuon`` together with the parameter groups, lr and weight decay:

    opt = PhysMuon(groups, lr=0.025, weight_decay=0.05, **language_model(total_steps))

    language_model(total_steps)          changes 1-5; the constants of the GPT-2 benchmark runs
    interatomic_potential(total_steps)   changes 1-4, parallel fast + slow with gated readout (w_max = 0.3)

Where each constant comes from, and how to adapt them to a new task, is in docs/hyperparameters.md. The potential
results of the paper were obtained with the deepmd-kit hooks in integrations/deepmd; ``interatomic_potential`` carries
the same constants to this package's optimizer.
"""
from __future__ import annotations


def language_model(total_steps: int = 3250) -> dict:
    """All five changes, with the constants of the GPT-2 124M benchmark runs; the readout switches at 23% of training."""
    return dict(momentum=0.95,
                bm_start=round(total_steps * 750 / 3250), bm_beta_fast=0.85, bm_beta_slow=0.985,   # 1
                iso=True, iso_lam=0.5, iso_clamp=2.0, iso_beta=0.95,                               # 2
                osc_gate=True, osc_wmax=0.6, osc_beta=0.9,                                          # 3
                bm_shadow=True,                                                                     # 4
                fresh_cap=0.25)                                                                     # 5


gpt_speedrun = language_model      # alias retained for compatibility


def interatomic_potential(total_steps: int, beta_fast: float = 0.90, beta_slow: float = 0.995,
                          w_max: float = 0.3, switch: float = 0.3077) -> dict:
    """Changes 1-4: unified interatomic-potential preset (parallel fast + slow modes with gated readout from
    ``switch`` on, fresh-share cap off).

    The defaults (``beta_fast=0.90, beta_slow=0.995, w_max=0.3``) are shared by all five 20,000-step and large-scale
    interatomic-potential tasks in the paper (DPA-3 on MPtrj, large-batch SPICE2 and ANI-1x; DPA4-Mini; DPA4-Pro).
    The 8,000-step regular-batch SPICE2 task uses ``beta_fast=0.85, beta_slow=0.98``.
    """
    return dict(bm_start=round(total_steps * switch), bm_beta_fast=beta_fast, bm_beta_slow=beta_slow,        # 1
                iso=True, iso_lam=0.5, iso_clamp=2.0, iso_beta=0.95,                                         # 2
                osc_gate=True, osc_wmax=w_max, osc_beta=0.9,                                                 # 3
                bm_shadow=True)                                                                              # 4


interatomic_potential_gated = interatomic_potential      # alias retained for compatibility


def paper_bm(total_steps: int = 3250) -> dict:
    """Change 1 alone with a fixed readout weight (the "+ 1" row of the GPT-2 results): mean memory 30 steps."""
    return dict(momentum=0.95, bm_start=round(total_steps * 1000 / 3250),
                bm_beta_fast=0.85, bm_beta_slow=0.98, bm_w_fast=0.4385)
