"""Freeze a DPA4 checkpoint for inference with ASE (`dp --pt freeze` without the multi-rank LAMMPS part, which needs
deepmd's custom operator library).

    python freeze.py model_ema.ckpt-1465000.pt model          # writes model.pt2

The frozen model is compiled for the type of GPU it is frozen on; freeze once per GPU type.
"""
import sys

from deepmd.pt.model.model.sezm_model import SeZMModel

SeZMModel.supports_edge_parallel = lambda self: False   # class level: freeze loads the model from the checkpoint itself
from deepmd.main import main  # noqa: E402

sys.argv = ["dp", "--pt", "freeze", "-c", sys.argv[1], "-o", sys.argv[2]]
main()
