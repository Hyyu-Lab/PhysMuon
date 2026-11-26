"""Minimal PhysMuon example: a small MLP fitted to synthetic data. Runs on a CPU in about a minute or less.

    python examples/quickstart.py                    # PhysMuon, all five changes
    python examples/quickstart.py --optimizer muon   # the same loop with every change off (plain Muon)
"""
import argparse
import pathlib
import sys

import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the repository root, if physmuon is not installed
from physmuon import PhysMuon, language_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["physmuon", "muon"], default="physmuon")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    d_in, width, d_out, batch, micro_batches = 32, 128, 8, 256, 4
    teacher = nn.Sequential(nn.Linear(d_in, 64), nn.Tanh(), nn.Linear(64, d_out)).requires_grad_(False)
    model = nn.Sequential(nn.Linear(d_in, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, d_out))
    loss_fn = nn.MSELoss()

    # weight matrices -> Muon with the five changes; everything else (here: the biases) -> AdamW inside the same optimizer
    matrices = [p for p in model.parameters() if p.ndim >= 2]
    others = [p for p in model.parameters() if p.ndim < 2]
    preset = language_model(args.steps) if args.optimizer == "physmuon" else {}
    opt = PhysMuon([dict(params=matrices), dict(params=others, use_muon=False, lr=3e-3)], lr=0.02, **preset)
    base_lrs = [g["lr"] for g in opt.param_groups]

    for step in range(args.steps):
        x = torch.randn(batch, d_in)
        y = teacher(x) + 0.1 * torch.randn(batch, d_out)
        total = 0.0
        for xb, yb in zip(x.chunk(micro_batches), y.chunk(micro_batches)):   # gradient accumulation over micro-batches
            loss = loss_fn(model(xb), yb) / micro_batches
            loss.backward()
            opt.observe_microbatch()                 # change 2 reads each micro-batch's gradient here
            total += float(loss.detach())
        frac = step / args.steps                     # constant learning rate, then a linear decay over the last 70%
        for group, base in zip(opt.param_groups, base_lrs):
            group["lr"] = base * (1.0 if frac < 0.3 else 1.0 - 0.9 * (frac - 0.3) / 0.7)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {total:.4f}")

    with torch.no_grad():
        x = torch.randn(4096, d_in)
        print(f"{args.optimizer}: held-out loss {float(loss_fn(model(x), teacher(x))):.4f}")


if __name__ == "__main__":
    main()
