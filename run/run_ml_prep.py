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

def run(defname, tag, feature_set, run="a"):

    # Output directory
    out_path = Path(f"../output/ml/veto/data/{run}/{tag}")
    out_path.mkdir(parents=True, exist_ok=True)

    # Processing 
    process = MLProcessor(
        defname=defname,
        # file_name="/exp/mu2e/data/users/sgrant/mu2e-cosmic/files/nts.mu2e.CosmicCRYSignalAllMix2BBTriggered.MDC2020aw_best_v1_3_v06_05_00.001202_00056492.root",
        feature_set=feature_set,
        use_remote=True # False, # False # True # False # True
    ) 
    results = process.execute()

    # Save results
    Write(out_path=out_path).write_all(results)
    
def main(): 

    configs = [
        # {
        #     defname: "nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aw_perfect_v1_3_v06_06_00.root",
        #     tag: "sig_onspill-LH_aw",
        # },
        # {
        #     "defname": "nts.mu2e.CeEndpointMix2BBTriggered.MDC2020aw_best_v1_3_v06_06_00.root",
        #     "tag": "sig_mix_onspill-LH_aw",
        #     "feature_set": "crv"
        # },
        {
            "defname": "nts.mu2e.CosmicCRYSignalAllOnSpillTriggered.MDC2020aw_perfect_v1_3_v06_06_00.root",
            "tag": "CRY_onspill-LH_aw",
            "feature_set": "crv",
        },
        {
            "defname": "nts.mu2e.CosmicCRYSignalAllMix2BBTriggered.MDC2020aw_best_v1_3_v06_06_00.root",
            "tag": "CRY_mix2BB_onspill-LH_aw",
            "feature_set": "crv"
        },
    ]

    
    for config in configs:
        run(config["defname"], config["tag"], config["feature_set"]) 

if __name__ == "__main__":
    main()