"""deepmd-kit command line with the PhysMuon optimizer hook of muon_pm_dp.py (environment variables ARM, PM, ISO, ...)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muon_pm_dp
muon_pm_dp.install()
from deepmd.main import main
if __name__ == "__main__":
    main()
