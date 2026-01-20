import pickle
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import os

from pyutils.pyprint import Print
from pyutils.pylogger import Logger
from pyutils.pyselect import Select
from pyutils.pycut import CutManager
from pyutils.pyplot import Plot
from pyutils.pyvector import Vector

import sys
sys.path.extend(["../src/utils"])
from io_manager import Load

# Make everything available when using "from preamble import *"
__all__ = ["Logger", "Print", "Load", "Draw"] 

pyutils_verb = 0
printer = Print()
selector = Select(verbosity=pyutils_verb)
logger = Logger(verbosity=pyutils_verb)
plotter = Plot(verbosity=pyutils_verb)
vector = Vector(verbosity=pyutils_verb)

"""
Assess the selection rate as a function of momentum with an without concrete sheidling

Strategy:
    0) Cosmic parent particle type distribution for selected tracks

    1) Plot selection rate for Run-1 and Run-1a as a function of momentum in all windows:
        - Cosmic parent momentum 
        - Track parent momentum 
        - Cosmic parent - track parent momentum 
        - Track reco momentum 

    2) Look at ratios of of Run-1 and Run-1a
        - Assess the slopes

    3) Fit the expected ratio per MeV for muons against energy loss 

Questions still up for grabs:
    1) What is the thickness of concrete?
    2) What type of concrete is it? 
    3) Based on this, what is the real expected energy loss? 
"""
@dataclass
class MomentumWindow:
 """Define momentum selection windows for track momentum"""
 name: str
 p_min: Optional[float] = None
 p_max: Optional[float] = None

 def apply(self, mom_mag):
     """Apply momentum cuts"""
     if self.p_min is None and self.p_max is None:
         return ak.ones_like(mom_mag, dtype=bool)
   
     mask = ak.ones_like(mom_mag, dtype=bool)
     if self.p_min is not None:
         mask = mask & (mom_mag > self.p_min)
     if self.p_max is not None:
         mask = mask & (mom_mag < self.p_max)
     return mask

class CosmicAna:
    """Pipeline for cosmic ray momentum analysis with livetime weighting
    """

    # Livetime for each window (seconds)
    LIVETIMES = {
        "inclusive": {"aw": 35189, "1a": 14954},
        "wide": {"aw": 28048, "1a": 11800},
        "extended": {"aw": 2941, "1a": 1218},
        "signal": {"aw": 383.89, "1a": 159.59},
    }
    
    # Momentum windows (GeV/c) applied to TRACK momentum at TT_Front
    WINDOWS = {
        "inclusive": MomentumWindow("inclusive"),
        "wide": MomentumWindow("wide", p_min=85, p_max=200),
        "extended": MomentumWindow("extended", p_min=100, p_max=110),
        "signal": MomentumWindow("signal", p_min=103.6, p_max=104.9),
    }
    
    # Plot styling
    DATASET_STYLES = {
        "aw": {"color": "red", "label": "Run-1 (MDC2020aw)"},
        "1a": {"color": "blue", "label": "Run-1a (MDC2025)"},
    }
    
    def __init__(self, data):
        self.data = data

    @classmethod
    def load_data(cls, base_path="../output/results"):
        """Load data from default paths"""
        return {
            "1a": Load(f"{base_path}/dev_CRY_onspill-LH_1a_noCRV").load_array_parquet(),
            "aw": Load(f"{base_path}/dev_CRY_onspill-LH_aw_noCRV").load_array_parquet(),
        }

    def select_by_window(self, data, window_name: str):
        """
        Select events based on track momentum window at TT_Front
        """
        # Get track momentum at TT_Front surface
        trk_front = selector.select_surface(data["trkfit"], "TT_Front")
        mom_mag = vector.get_mag(data["trkfit"]["trksegs"], "mom")
        mom_mag = mom_mag[trk_front]
        
        # Apply window cuts to track momentum
        window = self.WINDOWS[window_name]
        window_mask = window.apply(mom_mag)
        window_mask = ak.flatten(window_mask, axis=-1)
        
        # Apply mask to select events
        selected_data = {
            "trk": data["trk"][window_mask],
            "trkfit": data["trkfit"][window_mask],
            "trkmc": data["trkmc"][window_mask],
        }
        
        # Keep only events with tracks
        has_trks = ak.any(selected_data["trk"]["trk.pdg"], axis=-1)
        for key in selected_data:
            selected_data[key] = selected_data[key][has_trks]
        
        return selected_data
   
    def get_particle_composition(self, data: ak.Array, window_name: str) -> Dict: 
        
        from collections import Counter
        
        pdg_names = {
            11: "e-", -11: "e+",
            13: "mu-", -13: "mu+",
            22: "gamma",
            111: "pi0", 211: "pi+", -211: "pi-",
            221: "eta", 331: "eta'",
            130: "K_L0", 310: "K_S0", 321: "K+", -321: "K-",
            2112: "neutron", -2112: "antineutron",
            2212: "proton", -2212: "antiproton"
        } 

        composition = {}

        parents = self.get_cosmic_parents(data) 
        pdg_codes = ak.flatten(parents["pdg"], axis=None)
        pdg_codes_np = ak.to_numpy(pdg_codes)
        counts = Counter(pdg_codes_np)
        total = sum(counts.values())

        for code, label in pdg_names.items():
            composition[label] = counts[code]
        
        return composition

    def get_parent_mom(self, parents) -> ak.Array:
        """Extract momentum magnitude from parents"""
        mom_mag = vector.get_mag(parents, "mom")
        mom_mag = ak.firsts(mom_mag, axis=-1)
        mom_mag = ak.flatten(mom_mag, axis=None)
        return mom_mag
    
    def get_cosmic_parents(self, data, pdg: Optional[int] = None) -> ak.Array:
        """Get cosmic parents (rank == -1) with max momentum"""
        rank_mask = data["trkmc"]["trkmcsim"]["rank"] == -1
        mom = vector.get_mag(data["trkmc"]["trkmcsim"], "mom") #
        print("mom", mom)
        print("rank_mask1", rank_mask)
        max_mom_mask = mom == ak.max(mom, axis=-1)

        if pdg:
            pdg_mask = abs(data["trkmc"]["trkmcsim"]["pdg"]) == pdg
            condition = rank_mask & max_mom_mask & pdg_mask
        else:
            condition = rank_mask & max_mom_mask
        
        return data["trkmc"]["trkmcsim"][condition]
    
    def get_track_parents(self, data, pdg: Optional[int] = None) -> ak.Array:
        """Get track parents (max nhits)"""
        nhits_mask = data["trkmc"]["trkmcsim"]["nhits"] == \
                     ak.max(data["trkmc"]["trkmcsim"]["nhits"], axis=-1)
        
        if pdg:
            pdg_mask = abs(data["trkmc"]["trkmcsim"]["pdg"]) == pdg
            condition = nhits_mask & pdg_mask
        else:
            condition = nhits_mask
        
        return data["trkmc"]["trkmcsim"][condition]
    
    def get_cosmic_parent_mom(self, data, pdg: Optional[int] = None) -> ak.Array:
        """Extract cosmic parent momentum magnitudes"""
        cosmic_parents = self.get_cosmic_parents(data, pdg)
        return self.get_parent_mom(cosmic_parents)
    
    def get_track_parent_mom(self, data, pdg: Optional[int] = None) -> ak.Array:
        """Extract track parent momentum magnitudes"""
        track_parents = self.get_track_parents(data, pdg)
        return self.get_parent_mom(track_parents)

    def get_track_mom(self, data) -> ak.Array:
        """Extract track parent momentum magnitudes"""
        # Should already have the downstream electron selected
        at_trk_front = selector.select_surface(data["trkfit"], "TT_Front")
        mom_mag = vector.get_mag(data["trkfit"]["trksegs"][at_trk_front], "mom")
        mom_mag = ak.flatten(mom_mag, axis=None)
        return mom_mag 
    
    def get_weights(self, n_events: int, livetime: float) -> np.ndarray:
        """Create weights array: 1/livetime for each event"""
        return np.ones(n_events) / livetime
    
    def process_data(self, pdg: Optional[int] = None, windows: list = None) -> Dict:

        if windows is None:
            windows = list(self.WINDOWS.keys())
        
        results = {}

        for dataset_name, dataset in self.data.items():
            print(f"\nProcessing dataset: {dataset_name}")
            results[dataset_name] = {}

            # Apply each window
            for window_name in windows:
                print(f"\tApplying {window_name} window...")
                
                # Select events in window
                selected_data = self.select_by_window(dataset, window_name)

                # Find particle composition 
                composition = self.get_particle_composition(selected_data, window_name) 
                
                # Momenta 
                cosmic_parent_mom = self.get_cosmic_parent_mom(selected_data)
                track_parent_mom = self.get_track_parent_mom(selected_data)
                track_reco_mom = self.get_track_mom(selected_data)

                # Momenta for muons
                cosmic_parent_muon_mom = self.get_cosmic_parent_mom(selected_data, pdg=13)
                track_parent_muon_mom = self.get_track_parent_mom(selected_data, pdg=13)

                # Livetime (days)
                livetime = self.LIVETIMES[window_name][dataset_name]

                # Weights for rates
                cosmic_parent_weights = self.get_weights(len(cosmic_parent_mom), livetime)
                track_parent_weights = self.get_weights(len(track_parent_mom), livetime)
                track_reco_weights = self.get_weights(len(track_reco_mom), livetime)
                cosmic_parent_muon_weights = self.get_weights(len(cosmic_parent_muon_mom), livetime)
                track_parent_muon_weights = self.get_weights(len(track_parent_muon_mom), livetime)

                # Fill results 
                results[dataset_name][window_name] = {
                    "composition" : composition,
                     "cosmic_parent": {
                         "mom": ak.to_numpy(cosmic_parent_mom),
                         "weights": cosmic_parent_weights,
                         "n_events": len(cosmic_parent_mom),
                     },
                     "track_parent": {
                         "mom": ak.to_numpy(track_parent_mom),
                         "weights": track_parent_weights,
                         "n_events": len(track_parent_mom),
                     },
                     "cosmic_parent_muon": {
                         "mom": ak.to_numpy(cosmic_parent_muon_mom),
                         "weights": cosmic_parent_muon_weights,
                         "n_events": len(cosmic_parent_muon_mom),
                     },
                     "track_parent_muon": {
                         "mom":  ak.to_numpy(track_parent_muon_mom),
                         "weights": track_parent_muon_weights,
                         "n_events": len(track_parent_muon_mom),
                     },
                     "track_reco": {
                         "mom": ak.to_numpy(track_reco_mom),
                         "weights": track_reco_mom,
                         "n_events": len(track_reco_mom),
                     },
                     "livetime": livetime,
                }

        return results

    @classmethod
    def write_results(cls, results, file_name="shielding_ana.pkl"):
        out_path = Path("../output/misc/")
        # what's the syntax??
        # os.mkdirs(out_path, exist_ok=True, parents=True)
        out_name = out_path / file_name
        with open(out_name, "wb") as f:
            pickle.dump(results, f)
        print(f"\tWrote {out_name}")

    @classmethod
    def load_results(cls, file_name="shielding_ana.pkl"):
        in_path = Path("../output/misc/")
        in_name = in_path / file_name
        with open(in_name, "rb") as f:
            results = pickle.load(f)
        print(f"\tOpened {in_name}")
        return results 

    ########################
    # Plotting methods  
    ########################

    # Class attribute
    IMG_PATH = Path("../output/images/shielding_ana/")

    @classmethod
    def plot_particle_composition(cls, results):
        print("Plotting particle composition")
        for dataset_name, windows in results.items():
            fig, ax = plt.subplots(2, 2, figsize=(2*6.4, 2*4.8))
            ax = ax.flatten() 
       
            for i_window, (window_name, data) in enumerate(windows.items()):
                print(f"Window {window_name}: {i_window}")

                composition = data["composition"] 
                # print(f"ID of composition for {window_name}: {id(composition)}")
                
                labels, counts = [], []
                tot = sum(composition.values())
                for l, c in composition.items():
                    if c == 0:
                        continue
                    labels.append(l)
                    counts.append(100 * c / tot)
                
                #labels = list(composition.keys())
                #counts = list(composition.values())
                
                ax[i_window].barh(labels, counts) # , color='skyblue', edgecolor='black')
                ax[i_window].set_title(window_name.capitalize())
                ax[i_window].set_xlabel("Cosmic parents of selected tracks [%]")
                ax[i_window].invert_yaxis()  # Labels read top-to-bottom
                
            plt.tight_layout()
            out_name = cls.IMG_PATH / f"bar_{dataset_name}_particle_composition.png"
            plt.savefig(out_name)
            print(f"\tWrote {out_name}")
            plt.show()

###############
# Usage methods 
###############

def run_processing(ana):
    # Process data
    results = ana.process_data()
    # Cache results
    CosmicAna.write_results(results)

def run_plotting(ana):
    ana.IMG_PATH.mkdir(parents=True, exist_ok=True)
    results = ana.load_results()
    

    # Particle composition 
    ana.plot_particle_composition(results)

    # print(results)

def run(config=None):
    data = CosmicAna.load_data()
    ana = CosmicAna(data)
    if config == "process" or config is None:
        run_processing(ana)
    if config == "plot" or config is None:
        run_plotting(ana)

if __name__ == "__main__":
    # run() # run everything
    run("process") # run processing
    run("plot") # run plotting
