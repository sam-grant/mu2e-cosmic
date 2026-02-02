import pickle
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import os
from scipy import optimize 
from matplotlib.gridspec import GridSpec


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
    """Pipeline for cosmic ray momentum analysis with walltime weighting
    """

    ### Histograms

    # aw
    # 18598.1421205606
    # 18062.91723071614
    # 3014.7926122790254
    # 383.88543823326427

    # 1a
    # 7095.536959553696
    # 6879.8529356561
    # 1150.5373782524955
    # 159.5928226363009

    WALlTIMES = {
        "inclusive": {"aw": 18598.1421205606, "1a": 7095.536959553696},
        "wide": {"aw": 18062.91723071614, "1a": 6879.8529356561},
        "extended": {"aw": 3014.7926122790254, "1a": 1150.5373782524955},
        "signal": {"aw": 383.88543823326427, "1a": 159.5928226363009},
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
    
    def get_weights(self, n_events: int, walltime: float) -> np.ndarray:
        """Create weights array: 1/walltime for each event"""
        return np.ones(n_events) / walltime
    
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

                # Walltime 1BB (days)
                walltime = self.WALlTIMES[window_name][dataset_name]

                # Weights for rates
                cosmic_parent_weights = self.get_weights(len(cosmic_parent_mom), walltime)
                track_parent_weights = self.get_weights(len(track_parent_mom), walltime)
                track_reco_weights = self.get_weights(len(track_reco_mom), walltime)
                cosmic_parent_muon_weights = self.get_weights(len(cosmic_parent_muon_mom), walltime)
                track_parent_muon_weights = self.get_weights(len(track_parent_muon_mom), walltime)

                # Fill results 
                results[dataset_name][window_name] = {
                    "composition" : composition,
                     "cosmic_parent": {
                         "mom": ak.to_numpy(cosmic_parent_mom) * 1e-3, # GeV/c
                         "weights": cosmic_parent_weights,
                         "n_events": len(cosmic_parent_mom),
                     },
                     "track_parent": {
                         "mom": ak.to_numpy(track_parent_mom) * 1e-3, # GeV/c
                         "weights": track_parent_weights,
                         "n_events": len(track_parent_mom),
                     },
                     "cosmic_parent_muon": {
                         "mom": ak.to_numpy(cosmic_parent_muon_mom) * 1e-3, # GeV/c
                         "weights": cosmic_parent_muon_weights,
                         "n_events": len(cosmic_parent_muon_mom),
                     },
                     "track_parent_muon": {
                         "mom":  ak.to_numpy(track_parent_muon_mom) * 1e-3,
                         "weights": track_parent_muon_weights,
                         "n_events": len(track_parent_muon_mom),
                     },
                     "track_reco": {
                         "mom": ak.to_numpy(track_reco_mom),
                         "weights": track_reco_mom,
                         "n_events": len(track_reco_mom),
                     },
                     "walltime": walltime,
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
        
        # Create single figure for overlaid comparison
        fig, ax = plt.subplots(2, 2, figsize=(2*6.4, 2*4.8))
        ax = ax.flatten() 
        
        for i_window, window_name in enumerate(cls.WINDOWS.keys()):
            print(f"Window {window_name}: {i_window}")
            
            # Get composition for both datasets
            composition_1a = results["1a"][window_name]["composition"]
            composition_aw = results["aw"][window_name]["composition"]
            
            # Get all unique labels from both datasets
            all_labels = set(composition_1a.keys()) | set(composition_aw.keys())
            
            # Calculate percentages
            tot_1a = sum(composition_1a.values())
            tot_aw = sum(composition_aw.values())
            
            data_1a = {l: 100 * composition_1a.get(l, 0) / tot_1a for l in all_labels}
            data_aw = {l: 100 * composition_aw.get(l, 0) / tot_aw for l in all_labels}
            
            # Sort by average count (descending)
            labels_sorted = sorted(all_labels, 
                                key=lambda l: (data_1a[l] + data_aw[l]) / 2, 
                                reverse=True)
            
            # Remove labels with zero counts in both
            labels_sorted = [l for l in labels_sorted if data_1a[l] > 0 or data_aw[l] > 0]
            
            counts_1a = [data_1a[l] for l in labels_sorted]
            counts_aw = [data_aw[l] for l in labels_sorted]
            
            # Create positions for bars
            y_pos = np.arange(len(labels_sorted))
            height = 0.35
            
            # Plot bars side by side
            ax[i_window].barh(y_pos - height/2, counts_aw, height, 
                            label='Run-1', alpha=0.5)
            ax[i_window].barh(y_pos + height/2, counts_1a, height,
                            label='Run-1a', alpha=0.5)
            
            ax[i_window].set_yticks(y_pos)
            ax[i_window].set_yticklabels(labels_sorted)
            ax[i_window].set_title(window_name.capitalize())
            ax[i_window].set_xlabel("Cosmic parents of selected tracks [%]")
            ax[i_window].invert_yaxis()  # Labels read top-to-bottom
            ax[i_window].legend(loc="lower right")
            ax[i_window].tick_params(which="both", left=False, bottom=False) 
            
        plt.tight_layout()
        out_name = cls.IMG_PATH / f"bar_comparison_particle_composition.png"
        plt.savefig(out_name)
        print(f"\tWrote {out_name}")
        plt.show()

    @classmethod 
    def _format_label(cls, label):
        return label.replace("_", " ").capitalize()

    @classmethod
    def plot_ratios(cls, results, name="cosmic_parent", nbins=40, xrange=(0, 40), 
                    ylabel=None, xlabel=None, rate=False, norm=True, show=False):
        """
        Plots rate histograms using weights, calculates propagated errors,
        """
        fig, axes = plt.subplots(5, 2, figsize=(2*6.4, 2*4.8),  # Taller figure for better spacing
                            gridspec_kw={
                                'height_ratios': [2, 1, 1, 2, 1],
                                'hspace': 0.0,  # No gap between overlay and ratio
                                'wspace': 0.3, 
                                'left': 0.10, 'right': 0.98, 
                                'top': 0.97, 'bottom': 0.05
                            })

        xmin, xmax = xrange

        # Hide the spacer row axes
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')

        for i, window_name in enumerate(cls.WINDOWS.keys()):
            # Map to axes array
            col = i % 2
            if i < 2:  # Top row windows
                row_overlay = 0
                row_ratio = 1 
            else:  # Skip spacer
                row_overlay = 3
                row_ratio = 4  

            ax_overlay = axes[row_overlay, col]
            ax_ratio = axes[row_ratio, col]
            
            # Share x-axis between overlay and ratio
            ax_ratio.sharex(ax_overlay)

            # --- Data extraction ---
            mom_1a = results["1a"][window_name][name]["mom"]
            mom_aw = results["aw"][window_name][name]["mom"]

            if rate:
                weights_1a = results["1a"][window_name][name]["weights"]
                weights_aw = results["aw"][window_name][name]["weights"]
            else:
                weights_1a = np.ones_like(mom_1a)
                weights_aw = np.ones_like(mom_aw)

            # --- Weighted histogramming ---
            hist_aw, bins = np.histogram(mom_aw, bins=nbins, range=xrange, weights=weights_aw)
            hist_1a, _ = np.histogram(mom_1a, bins=nbins, range=xrange, weights=weights_1a)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Error = sqrt(sum of weights^2)
            err_sq_aw, _ = np.histogram(mom_aw, bins=nbins, range=xrange, weights=weights_aw**2)
            err_sq_1a, _ = np.histogram(mom_1a, bins=nbins, range=xrange, weights=weights_1a**2)
            err_aw, err_1a = np.sqrt(err_sq_aw), np.sqrt(err_sq_1a)

            if norm:
                # normalise histograms AND errors
                n_aw = np.sum(hist_aw)
                n_1a = np.sum(hist_1a)
                hist_aw = hist_aw / n_aw
                hist_1a = hist_1a / n_1a
                err_aw = err_aw / n_aw
                err_1a = err_1a / n_1a

            # --- Ratio and error propagation ---
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = hist_1a / hist_aw
                ratio_err = ratio * np.sqrt((err_1a / np.where(hist_1a > 0, hist_1a, 1))**2 +
                                            (err_aw / np.where(hist_aw > 0, hist_aw, 1))**2)

            if rate:
                label_1a = f"Run-1a: {np.sum(weights_1a):.3f}/day"
                label_aw = f"Run-1: {np.sum(weights_aw):.3f}/day"
            else:
                label_1a = f"Run-1a" # : {np.sum(weights_1a):.3f}/day"
                label_aw = f"Run-1" # : {np.sum(weights_aw):.3f}/day"
                
            # --- Overlay plot ---
            ax_overlay.step(bin_centers, hist_aw, where='mid', color='blue', 
                            label=label_aw)
            ax_overlay.step(bin_centers, hist_1a, where='mid', color='red', 
                            label=label_1a) # f'Run-1a: {np.sum(weights_1a):.3f}/day')
            
            # Error bands
            ax_overlay.fill_between(bin_centers, hist_aw - err_aw, hist_aw + err_aw, 
                                    color='blue', alpha=0.2, step='mid')
            ax_overlay.fill_between(bin_centers, hist_1a - err_1a, hist_1a + err_1a, 
                                    color='red', alpha=0.2, step='mid')

            # ax_overlay.set_yscale('log')
            ax_overlay.set_ylabel(ylabel) # f"Rate [tracks/day]") # / {unit}")
            ax_overlay.set_title(f"{window_name.capitalize()} window", fontsize=18)
            ax_overlay.legend(fontsize=18)
            ax_overlay.tick_params(labelbottom=False, direction='in', which='both', right=True, top=True)

            # --- Fitting ---
            mask = np.isfinite(ratio) & (ratio_err > 0) & (hist_aw > 0)
            xf, rf, erf = bin_centers[mask], ratio[mask], ratio_err[mask]

            # if len(xf) > 2:
            #     def line(x, m, b): return m * x + b
            #     try:
            #         popt, pcov = optimize.curve_fit(line, xf, rf, sigma=erf, absolute_sigma=True)
            #         m, b = popt
            #         m_err = np.sqrt(pcov[0, 0])
            #         c_err = np.sqrt(pcov[1, 1])
                    
            #         red_chi2 = np.sum(((rf - line(xf, *popt)) / erf)**2) / (len(xf) - 2)

            #         ax_ratio.plot(xf, line(xf, *popt), 'r-', lw=2)

            #         stats_str = f"Gradient: {m:.4f}"+r" $\pm$ "+f"{m_err:.4f}" 
            #         # stats_str += "\nIntercept: {c:.4f}"+r" $\pm$ "+f"{c_err:.4f}"
            #         # stats_str += r"$\chi^2/\text{ndf}$"+f": {red_chi2:.2f}"

            #         ax_ratio.text(0.95, 0.9, stats_str, transform=ax_ratio.transAxes,
            #                     verticalalignment='top',horizontalalignment='right', 
            #                     bbox=dict(facecolor='white', alpha=.7, edgecolor='none'))
            #     except:
            #         pass

            # --- Ratio plot formatting ---
            ax_ratio.errorbar(xf, rf, yerr=erf, fmt='ko', markersize=5)
            ax_ratio.axhline(y=1, color='gray', linestyle='--', linewidth=1)
            ax_ratio.set_ylabel("1a / aw", loc="center")
            ax_ratio.set_xlabel(f"{cls._format_label(name)} momentum [GeV/c]")
            ax_ratio.set_xlim(xmin, xmax)
            ax_ratio.set_ylim(0, 4)
            ax_ratio.tick_params(direction='in', which='both', right=True, top=True)

        # Align y-labels (skip spacer row)
        fig.align_ylabels([axes[0, 0], axes[1, 0], axes[3, 0], axes[4, 0]])
        fig.align_ylabels([axes[0, 1], axes[1, 1], axes[3, 1], axes[4, 1]])

        suffix = "rate" if rate else ("norm" if norm else "count")
        out_name = cls.IMG_PATH / f"hratios_{name}_{suffix}.png"
        plt.savefig(out_name, dpi=300, bbox_inches='tight')  # bbox_inches helps with spacing
        print(f"Wrote {out_name}")
        if show: 
            plt.show()
        plt.close(fig)

    @classmethod
    def plot_and_fit_ratios(cls, results, name="cosmic_parent", nbins=40, xrange=(0, 40), show=False):
        """
        Plot momentum histograms using weights, calculate propagated errors,
        fit a power law model to ratio
        """
        fig, axes = plt.subplots(5, 2, figsize=(2*6.4, 2*4.8),  # Taller figure for better spacing
                            gridspec_kw={
                                'height_ratios': [2, 1, 1, 2, 1],
                                'hspace': 0.0,  # No gap between overlay and ratio
                                'wspace': 0.3, 
                                'left': 0.10, 'right': 0.98, 
                                'top': 0.97, 'bottom': 0.05
                            })

        xmin, xmax = xrange

        # Hide the spacer row axes
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')

        for i, window_name in enumerate(cls.WINDOWS.keys()):
            # Map to axes array
            col = i % 2
            if i < 2:  # Top row windows
                row_overlay = 0
                row_ratio = 1 
            else:  # Skip spacer
                row_overlay = 3
                row_ratio = 4  

            ax_overlay = axes[row_overlay, col]
            ax_ratio = axes[row_ratio, col]
            
            # Share x-axis between overlay and ratio
            ax_ratio.sharex(ax_overlay)

            # --- Data extraction ---
            mom_1a = results["1a"][window_name][name]["mom"]
            mom_aw = results["aw"][window_name][name]["mom"]

            # Rates
            weights_1a = results["1a"][window_name][name]["weights"]
            weights_aw = results["aw"][window_name][name]["weights"]

            # --- Weighted histogramming ---
            hist_aw, bins = np.histogram(mom_aw, bins=nbins, range=xrange, weights=weights_aw)
            hist_1a, _ = np.histogram(mom_1a, bins=nbins, range=xrange, weights=weights_1a)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Error = sqrt(sum of weights^2)
            err_sq_aw, _ = np.histogram(mom_aw, bins=nbins, range=xrange, weights=weights_aw**2)
            err_sq_1a, _ = np.histogram(mom_1a, bins=nbins, range=xrange, weights=weights_1a**2)
            err_aw, err_1a = np.sqrt(err_sq_aw), np.sqrt(err_sq_1a)

            # Normalise histograms AND errors
            # n_aw = np.sum(hist_aw)
            # n_1a = np.sum(hist_1a)
            # hist_aw = hist_aw / n_aw
            # hist_1a = hist_1a / n_1a
            # err_aw = err_aw / n_aw
            # err_1a = err_1a / n_1a

            # --- Ratio and error propagation ---
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = hist_1a / hist_aw
                ratio_err = ratio * np.sqrt((err_1a / np.where(hist_1a > 0, hist_1a, 1))**2 +
                                            (err_aw / np.where(hist_aw > 0, hist_aw, 1))**2)

            # --- Overlay plot ---
            ax_overlay.step(bin_centers, hist_aw, where='mid', color='blue', 
                            label="Run-1")
            ax_overlay.step(bin_centers, hist_1a, where='mid', color='red', 
                            label="Run-1a")
            
            # Error bands
            ax_overlay.fill_between(bin_centers, hist_aw - err_aw, hist_aw + err_aw, 
                                    color='blue', alpha=0.2, step='mid')
            ax_overlay.fill_between(bin_centers, hist_1a - err_1a, hist_1a + err_1a, 
                                    color='red', alpha=0.2, step='mid')

            # ax_overlay.set_yscale('log')
            ax_overlay.set_ylabel("Rate [tracks/day]")
            ax_overlay.set_title(f"{window_name.capitalize()} window", fontsize=18)
            ax_overlay.legend(fontsize=18)
            ax_overlay.tick_params(labelbottom=False, direction='in', which='both', right=True, top=True)

            # --- Fitting ---
            mask = np.isfinite(ratio) & (ratio_err > 0) & (hist_aw > 0)
            xf, rf, erf = bin_centers[mask], ratio[mask], ratio_err[mask]

            # Power law function 
            # N_1a / N_aw = ( (p_0 + ∆p) / p_0)^1.7
            # So p_0 is the momentum threshold
            # ∆p is the offset for momentum loss ~= 420 MeV/c
            # We fit for ∆p 

            if len(xf) > 2:
                def power_law(x, delta_p): # , gamma): 
                    return ((x + delta_p)/x) ** 1.7 # gamma
                try:
                    popt, pcov = optimize.curve_fit(power_law, xf, rf, sigma=erf, absolute_sigma=True)
                    print("popt", popt)
                    print("pcov", pcov)
                    delta_p = popt[0]
                    # gamma = popt[1]
                    # print(p)
                    delta_p_err = np.sqrt(pcov[0, 0])
                    # gamma_err = np.sqrt(pcov[1, 1])
                    chi2_ndf = np.sum(((rf - power_law(xf, *popt)) / erf)**2) / (len(xf) - 2)
                    print("chi2", chi2_ndf)
                    ax_ratio.plot(xf, power_law(xf, *popt), 'r-', lw=2)

                    stats_str = (
                        r"$\Delta p = "
                        + f"{delta_p*1e3:.0f}"
                        + r" \pm "
                        + f"{delta_p_err*1e3:.0f}"
                        + r"\ \mathrm{MeV}/c$"
                        # + "\n"
                        # + r"$\gamma$ ="
                        # + f"{gamma:.0f}"
                        # + r" \pm "
                        # + f"{gamma_err:.0f}"
                        "\n"
                        + r"$\quad \chi^2/\mathrm{ndf} = "
                        + f"{chi2_ndf:.2f}"
                        + r"$"
                    )

                    # stats_str += "\nIntercept: {c:.4f}"+r" $\pm$ "+f"{c_err:.4f}"
                    # stats_str += r"$\chi^2/\text{ndf}$"+f": {red_chi2:.2f}"

                    ax_ratio.text(0.95, 0.9, stats_str, transform=ax_ratio.transAxes,
                                verticalalignment='top',horizontalalignment='right', 
                                bbox=dict(facecolor='white', alpha=.7, edgecolor='none'))
                except Exception as e:
                    print(f"Error fitting power law: {e}")
                    pass 

            # --- Ratio plot formatting ---
            ax_ratio.errorbar(xf, rf, yerr=erf, fmt='ko', markersize=5)
            ax_ratio.axhline(y=1, color='gray', linestyle='--', linewidth=1)
            ax_ratio.set_ylabel("1a / aw", loc="center")
            ax_ratio.set_xlabel(f"{cls._format_label(name)} momentum [GeV/c]")
            ax_ratio.set_xlim(xmin, xmax)
            ax_ratio.set_ylim(0, 4)
            ax_ratio.tick_params(direction='in', which='both', right=True, top=True)

        # Align y-labels (skip spacer row)
        fig.align_ylabels([axes[0, 0], axes[1, 0], axes[3, 0], axes[4, 0]])
        fig.align_ylabels([axes[0, 1], axes[1, 1], axes[3, 1], axes[4, 1]])

        suffix = "power_law" #  if rate else ("norm" if norm else "count")
        out_name = cls.IMG_PATH / f"hratios_{name}_{suffix}.png"
        plt.savefig(out_name, dpi=300, bbox_inches='tight')  # bbox_inches helps with spacing
        print(f"Wrote {out_name}")
        if show: 
            plt.show()
        plt.close(fig)
    
    @classmethod
    def histogram_ratios(cls, results, window_name="extended", name="cosmic_parent", 
                        nbins=60, xrange=(-0.5, 5.5), show=False):
        """
        Create histograms of rate ratios for a single window
        Shows all momentum, p < 5 GeV/c, and p > 5 GeV/c separately
        """
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 3*4.8))
        
        # --- Data extraction ---
        mom_1a = results["1a"][window_name][name]["mom"]
        mom_aw = results["aw"][window_name][name]["mom"]
        
        weights_1a = results["1a"][window_name][name]["weights"]
        weights_aw = results["aw"][window_name][name]["weights"]
        
        # --- Weighted histogramming ---
        hist_aw, bins = np.histogram(mom_aw, bins=nbins, range=xrange, weights=weights_aw)
        hist_1a, _ = np.histogram(mom_1a, bins=nbins, range=xrange, weights=weights_1a)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Error = sqrt(sum of weights^2)
        err_sq_aw, _ = np.histogram(mom_aw, bins=nbins, range=xrange, weights=weights_aw**2)
        err_sq_1a, _ = np.histogram(mom_1a, bins=nbins, range=xrange, weights=weights_1a**2)
        err_aw, err_1a = np.sqrt(err_sq_aw), np.sqrt(err_sq_1a)
        
        # --- Ratio and error propagation ---
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = hist_1a / hist_aw
            ratio_err = ratio * np.sqrt((err_1a / np.where(hist_1a > 0, hist_1a, 1))**2 +
                                        (err_aw / np.where(hist_aw > 0, hist_aw, 1))**2)
        
        # Filter valid ratios
        mask = np.isfinite(ratio) & (ratio_err > 0) & (hist_aw > 0)
        valid_mom = bin_centers[mask]
        valid_ratio = ratio[mask]
        valid_ratio_err = ratio_err[mask]
        
        # --- Three regions ---
        regions = [
            ("All momentum", lambda p: np.ones_like(p, dtype=bool), 0),
            ("p < 5 GeV/c", lambda p: p < 5.0, 1),
            ("p > 5 GeV/c", lambda p: p >= 5.0, 2)
        ]
        
        for region_name, region_mask_func, idx in regions:
            ax = axes[idx]
            region_mask = region_mask_func(valid_mom)
            
            if np.sum(region_mask) == 0:
                ax.text(0.5, 0.5, f"No data in range", 
                    transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f"{region_name}")
                continue
            
            region_ratio = valid_ratio[region_mask]
            region_ratio_err = valid_ratio_err[region_mask]
            
            # Calculate mean ratio and uncertainty
            # Weighted mean
            weights = 1 / (region_ratio_err**2)
            mean_ratio = np.sum(region_ratio * weights) / np.sum(weights)
            mean_ratio_err = 1 / np.sqrt(np.sum(weights))
            
            # Histogram of ratio values
            ax.hist(region_ratio, bins=30, range=(0, 4), 
                alpha=0.7, color='steelblue', edgecolor='black')
            
            # Add vertical line for mean
            ax.axvline(mean_ratio, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_ratio:.3f} ± {mean_ratio_err:.3f}')
            ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.5,
                    label='Unity')
            
            ax.set_xlabel("Rate ratio (1a / aw)")
            ax.set_ylabel("Number of bins")
            ax.set_title(f"{region_name}")
            ax.legend()
            # ax.grid(alpha=0.3)
            
            # Add text box with statistics
            stats_text = (
                f"N: {np.sum(region_mask)}\n"
                f"Mean: {mean_ratio:.3f} ± {mean_ratio_err:.3f}\n"
                f"Std Dev: {np.std(region_ratio):.3f}"
            )
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        fig.suptitle(f"{window_name.capitalize()} window", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        out_name = cls.IMG_PATH / f"hratio_dist_{window_name}_{name}.png"
        plt.savefig(out_name, dpi=300)
        print(f"Wrote {out_name}")
        
        if show:
            plt.show()
        plt.close(fig)

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

    ana.plot_particle_composition(results)

    ana.plot_ratios(results, xrange=(-0.5,20.5), nbins=21,
                    rate=False, norm=True, show=True,
                    ylabel="Tracks [normalized]"
    )

    ana.plot_ratios(results, xrange=(-0.5,20.5), nbins=21,
                    rate=True, norm=False, show=True,
                    ylabel="Rate [tracks/day]"
    )
    
    ana.plot_and_fit_ratios(results, xrange=(-0.5,20.5), nbins=21, show=True)

    # Something simpler: overlay histograms of the rate ratios for one window
    # All momentum 
    # Below 5 GeV/c
    # Above 5 GeV/c
    ana.histogram_ratios(results, window_name="extended", xrange=(-0.5, 5.5), nbins=60, show=True)

    # ana.plot_and_fit_ratios_2(resiults, xrange=(-0.5,10.5), nbins=22, show=True)

    # ana.plot_sig_ratio(results, xrange=(-0.5,20.5), nbins=21)
      # nbins=80)

    # ana.plot_sig_ratio_pred(results, xrange=(-0.5,20.5), nbins=21)

def run(config=None):
    data = CosmicAna.load_data()
    ana = CosmicAna(data)
    if config == "process" or config is None:
        run_processing(ana)
    if config == "plot" or config is None:
        run_plotting(ana)

if __name__ == "__main__":
    # run() # run everything
    # run("process") # run processing
    run("plot") # run plotting
