# System tools  
import os
import sys
from pathlib import Path

# Python stack 
import pandas as pd
import awkward as ak

# pyutils
from pyutils.pylogger import Logger

# ML tools
from sklearn.model_selection import train_test_split

# Internal modules 
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(script_dir, "..", "utils"),
    os.path.join(script_dir, "..", "core")
])
from io_manager import Load

class LoadML():
    """ Load up ML dataset """
    def __init__(self, run="h", verbosity=1):
        self.run = run
        self.base_in_path = Path(f"../../output/ml/{self.run}/data/")
        self.logger = Logger(print_prefix="[LoadML]", verbosity=verbosity)
        self.logger.log("Initialised", "success")

    def get_results(self):
        """ Raw full processing output from pkl files
        Can use be used seperately for validation
        without cluttering main ML notebook """ 
        # Define input paths
        cry_in_path = self.base_in_path / "CRY_onspill-LH_aw"
        ce_mix_in_path = self.base_in_path/ "CE_mix2BB_onspill-LH_aw"

        # Load data
        cry_data = Load(in_path=cry_in_path).load_pkl()
        ce_mix_data = Load(in_path=ce_mix_in_path).load_pkl()

        self.logger.log("Got full results", "success") 

        return cry_data, ce_mix_data

