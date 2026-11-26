# Trained models

The DPA4-Pro models trained with PhysMuon on MPtrj and evaluated on Matbench Discovery. Each file is a deepmd-kit
(PyTorch backend) checkpoint holding the exponential moving average of the weights, about 91 MB.

- `model_ema.ckpt-1465000.pt`: 1,465,000 training steps, the model of the main result (F1 0.8712, κ_SRME 0.1860,
  RMSD_AUC 0.7734, CPS 0.8757).
- `model_ema.ckpt-1300000.pt`, `model_ema.ckpt-1100000.pt`, and `model_ema.ckpt-900000.pt`: earlier checkpoints reported
  in the paper (CPS 0.8727 at 1,300,000 steps, 0.8698 at 1,100,000 steps, and 0.8654 at 900,000 steps).

The model files are not stored in git. The 900k-step and 1.465M-step checkpoints are publicly available on Hugging Face
at [orange4664/DPA-4.0.1-Pro-MPtrj-Physmuon](https://huggingface.co/orange4664/DPA-4.0.1-Pro-MPtrj-Physmuon). Downloaded
files placed in this folder can be verified against `SHA256SUMS` with `sha256sum -c SHA256SUMS` (for the 1.1M, 1.3M,
and 1.465M checkpoints). A checkpoint is frozen for inference with deepmd-kit 3.2.0:

```bash
dp --pt freeze -c model_ema.ckpt-1465000.pt -o dpa4pro_physmuon
```

The evaluation of the paper froze the checkpoints with [`../freeze.py`](../freeze.py); the evaluation steps and the
metrics of the three checkpoints are in [`../README.md`](../README.md).
