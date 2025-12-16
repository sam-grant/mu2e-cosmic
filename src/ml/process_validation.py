# System tools  
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore") # suppress warnings

# Internal modules 
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(script_dir, "..", "utils"),
    os.path.join(script_dir, "..", "core")
])
from draw import Draw
from load import LoadML

class ProcessValidation():
    """ Load up ML dataset """
    def __init__(self, run="h"):
        self.run = run
        self.base_in_path = Path(f"../../output/ml/{self.run}/data/")
        # A bit hacky but...
        # this is defined from the notebook/ml location
        self.img_out_path = Path(f"../../../output/images/ml/{self.run}/process/") 
        self.cry_data, self.ce_mix_data = LoadML(run=self.run).get_full_results()

    def draw_hists(self):
        draw = Draw()
        draw.plot_summary(self.cry_data["hists"], out_path = self.img_out_path / "h1o_3x3_cuts_CRY.png")
        draw.plot_summary(self.ce_mix_data["hists"], out_path = self.img_out_path / "h1o_3x3_cuts_CE_mix.png")
        
    def get_cut_flows(self):
        return self.cry_data["cut_flow"], self.ce_mix_data["cut_flow"]