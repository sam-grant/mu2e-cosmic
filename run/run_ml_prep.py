"""
Prepare cosmic data for ML work
Sam Grant 2025
"""

# preamble
import sys
from pathlib import Path

sys.path.extend(["../src/core", "../src/utils"])

from ml_process import MLProcessor
from draw import Draw
from io_manager import Write

def run_ml_prep():
    
    # Inputs...
    # signal_file = "../input/files/nts.mu2e.CosmicCRYSignalAllOnSpillTriggered.MDC2020aw_perfect_v1_3_v06_06_00.001202_00042784.root"
    # background_file = "../input/files/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aw_perfect_v1_3_v06_06_00.001210_00000580.root"
    signal_defname = "nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aw_perfect_v1_3_v06_06_00.root"
    background_defname = "nts.mu2e.CosmicCRYSignalAllOnSpillTriggered.MDC2020aw_perfect_v1_3_v06_06_00.root"

    # nts.mu2e.CosmicCRYSignalAllMix2BBTriggered.MDC2020aw_best_v1_3_v06_06_00.root
    # nts.mu2e.CeEndpointMix2BBTriggered.MDC2020aw_best_v1_3_v06_06_00.root

    # Output directories
    signal_out_path = Path("../output/ml/data/b/sig")
    background_out_path = Path("../output/ml/data/b/bkg")
    # Create directories (including parents) if they don't exist
    signal_out_path.mkdir(parents=True, exist_ok=True)
    background_out_path.mkdir(parents=True, exist_ok=True)
    
    # Processing 
    results = {} 
    
    # Signal
    process = MLProcessor(
        # file_name=signal_file,
        defname=signal_defname
    ) 
    results["signal"] = process.execute()
    
    # Background
    process = MLProcessor(
        # file_name=background_file, 
        defname=background_defname
    ) 
    results["background"] = process.execute()
    
    # save results
    Write(out_path=signal_out_path).write_all(results["signal"])
    Write(out_path=background_out_path).write_all(results["background"])

def main(): 
    run_ml_prep()

if __name__ == "__main__":
    main()