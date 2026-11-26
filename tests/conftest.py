import pathlib
import sys

here = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent)]          # tests/ (for the reference) and the repository root
