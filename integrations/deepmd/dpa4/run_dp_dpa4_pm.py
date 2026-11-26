"""deepmd-kit command line with PhysMuon patched into HybridMuon (pm_hybrid.py; environment variables ARM=hmuon|bm, BM_SCHED, PM, ISO, ...)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pm_hybrid
pm_hybrid.install()
if os.environ.get("VAL_FREE", "0") == "1":
    # full validation runs on rank 0 only and leaves ~40 GB in torch's cache (80.5 of 81 GB used), so the
    # checkpoint save right after it can OOM; return the cache before and after each validator pass and before each save.
    # Evaluation and training numerics are unchanged.
    import torch
    from deepmd.pt_expt.train.validation import FullValidator
    _run = FullValidator.run

    def _run_free(self, *args, save_checkpoint=None, **kwargs):
        if not self.should_run(kwargs["display_step"]):  # run() is called every step; only due steps pay
            return _run(self, *args, save_checkpoint=save_checkpoint, **kwargs)
        torch.cuda.empty_cache()
        def _save(*a, **k):
            torch.cuda.empty_cache()
            return save_checkpoint(*a, **k)
        out = _run(self, *args, save_checkpoint=_save if save_checkpoint is not None else None, **kwargs)
        torch.cuda.empty_cache()
        return out

    FullValidator.run = _run_free
    print("[run_dp] VAL_FREE=1: empty_cache before/after each full-validation pass and before its checkpoint saves", flush=True)
from deepmd.main import main
if __name__ == "__main__":
    main()
