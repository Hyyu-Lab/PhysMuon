from .optim import PhysMuon, newton_schulz, newton_schulz5
from .recipes import (gpt_speedrun, interatomic_potential, interatomic_potential_gated, language_model, paper_bm)
from .schedule import age, beta_for_age, kernel_age, fast_weight_for_age, switch_step

__all__ = ["PhysMuon", "newton_schulz", "newton_schulz5",
           "language_model", "interatomic_potential", "interatomic_potential_gated", "gpt_speedrun", "paper_bm",
           "age", "beta_for_age", "kernel_age", "fast_weight_for_age", "switch_step"]
