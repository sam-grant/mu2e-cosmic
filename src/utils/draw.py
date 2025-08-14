import os
import matplotlib.pyplot as plt
# Get the directory where draw.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Style file is in the same directory
style_path = os.path.join(current_dir, "mu2e.mplstyle")
# Use style
plt.style.use(style_path)
from matplotlib.ticker import ScalarFormatter

from analyse import Analyse
from pyutils.pylogger import Logger

class Draw():
    """
    Class to draw standard "hist" histograms produced by Analyse().create_histograms()
    """
    def __init__(self, cutset_name="alpha", on_spill=False, verbosity=1): 
        """
        Initialise 
        
        Args:
            cutset_name (str, opt): The cutset name. Defaults to 'alpha'.
            on_spill (bool, opt): Whether we are using onspill time cuts. Defaults to False.
            verbosity (int, optional): Level of output detail (0: critical errors only, 1: info, 2: debug, 3: deep debug)
        """
        # Verbosity 
        self.verbosity = verbosity
        # Start logger
        self.logger = Logger(
            print_prefix="[Plot]",
            verbosity=self.verbosity
        )
        # onspill 
        self.on_spill = on_spill
        # Get analysis params
        self.analyse = Analyse(cutset_name=cutset_name, verbosity=0)
        # Confirm
        self.logger.log(f"Initialised", "info")
        
    def _count_events(self, hists, selection, label):
        """
        Utility to count events from hist selections
        """
        h = hists[selection] 
        h = h[{"selection": label}]  
        return int(h.sum())
    
    def plot_mom_windows(self, hists, out_path=None):
        """
        Plot 1x3 momentum window histograms
        """
        # 1x3 subplots
        fig, ax = plt.subplots(1, 3, figsize=(6.4*3, 4.8))
        
        # Nested helper
        def _format_axis(ax, labels, title): 
            # Extend y-axis for legend space
            # ylim = ax.get_ylim()
            # ax.set_ylim(ylim[0], ylim[1]) # * 10)  # Extend top by factor of 10 (doesn't work)
            ax.set_yscale("log")
            ax.set_title(title)
            ax.legend(labels, frameon=True, loc="upper right")

        # Wide range 
        name = "mom_full"
        h1o_mom_wide = hists[name]
        # h_wide = h_wide[{"selection": ["All", "CE-like", "Unvetoed CE-like"]}] # slice(-2, None)}]  # Last 2 selections
        h1o_mom_wide.plot1d(overlay="selection", ax=ax[0], yerr=False)
        # Get hist labels
        labels = list(h1o_mom_wide.axes["selection"])
        for i, label in enumerate(labels):
            labels[i] = f"{label}: {"{:,}".format(self._count_events(hists, name, label))}"
        # Format
        title = "Wide range: 0-100 MeV/c"
        _format_axis(ax[0], labels, title)
    
        # Extended window 
        name = "mom_ext"
        h1o_mom_ext = hists[name]
        h1o_mom_ext.plot1d(overlay="selection", ax=ax[1], yerr=False)
        # Get hist labels
        labels = list(h1o_mom_ext.axes["selection"])
        for i, label in enumerate(labels):
            labels[i] = f"{label}: {"{:,}".format(self._count_events(hists, name, label))}"
        # Format
        title = f"Extended window: {self.analyse.thresholds["lo_ext_win_mevc"]}"\
                f"-{self.analyse.thresholds["hi_ext_win_mevc"]} MeV/c"
        _format_axis(ax[1], labels, title)
        
        # Signal window 
        name = "mom_sig"
        h1o_mom_sig = hists[name]
        h1o_mom_sig.plot1d(overlay="selection", ax=ax[2], yerr=False)
        # Get hist labels
        labels = list(h1o_mom_sig.axes["selection"])
        for i, label in enumerate(labels):
            labels[i] = f"{label}: {"{:,}".format(self._count_events(hists, name, label))}"
        title = f"Signal window: {self.analyse.thresholds["lo_sig_win_mevc"]}"\
                f"-{self.analyse.thresholds["hi_sig_win_mevc"]} MeV/c"
        # Format
        _format_axis(ax[2], labels, title) 
        
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=300)
            self.logger.log(f"\tWrote {out_path}", "success")
        plt.show()

    def plot_crv_z(self, hists, out_path=None):
        """
        Plot CRV z-position histograms
        """
        selection = ["All", "Preselect", "CE-like"]
        # 1x3 subplots
        fig, ax = plt.subplots() 
        # Wide range 
        name = "crv_z"
        h1o_z = hists[name]
        h1o_z = h1o_z[{"selection": selection}]
        h1o_z.plot1d(overlay="selection", density=True, ax=ax, yerr=False)
        # Format
        # ax.grid(True, alpha=0.3)
        ax.set_ylabel("Normalised coincidences")
        ax.legend(selection, frameon=False, loc="upper right")
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0,0), useMathText=True)
    
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=300)
            self.logger.log(f"\tWrote {out_path}", "success")
        plt.show()

    # def plot_trk_params(self, hists, out_path=None):
    #     """
    #     Plot 1x2 track-level histograms
    #     """
    #     # 1x3 subplots
    #     fig, ax = plt.subplots(1, 2, figsize=(6.4*2, 4.8))

    #     selection = ["All", "Preselect", "CE-like"]
        
    #     # Nested helper
    #     # Shouldn't this be a regular function?
    #     def _format_axis(ax, labels): 
    #         ax.set_yscale("log")
    #         ax.legend(labels, frameon=True, loc="upper right")

    #     # trkqual 
    #     name = "trkqual"
    #     h1o_trkqual = hists[name]
    #     # h_wide = h_wide[{"selection": ["All", "CE-like", "Unvetoed CE-like"]}] # slice(-2, None)}]  # Last 2 selections
    #     h1o_trkqual = h1o_trkqual[{"selection": selection}]
    #     h1o_trkqual.plot1d(overlay="selection", ax=ax[0], density=False, yerr=False)
    #     # Get hist labels
    #     labels = list(h1o_trkqual.axes["selection"])
    #     # for i, label in enumerate(labels):
    #     #     labels[i] = f"{label}: {"{:,}".format(self._count_events(hists, name, label))}"
    #     # Format
    #     # title = "Wide range: 0-300 MeV/c"
    #     _format_axis(ax[0], labels)
    
    #     # nactive 
    #     name = "nactive"
    #     h1o_nactive = hists[name]
    #     h1o_nactive = h1o_nactive[{"selection": selection}]
    #     h1o_nactive.plot1d(overlay="selection", ax=ax[1], yerr=False)
    #     # Get hist labels
    #     labels = list(h1o_nactive.axes["selection"])
    #     # for i, label in enumerate(labels):
    #     #     labels[i] = f"{label}: {"{:,}".format(self._count_events(hists, name, label))}"
    #     # Format
    #     _format_axis(ax[1], labels)
        
    #     plt.tight_layout()
    #     if out_path:
    #         plt.savefig(out_path, dpi=300)
    #         self.logger.log(f"\tWrote {out_path}", "success")
    #     plt.show()

    def plot_summary(self, hists, out_path=None, toggle_lines=None):
        """
        Plot 3x3 summary plot for ["All", "Preselect", "CE-like"]
        mom_full, pz, t0,
        trkqual, nactive, t0err
        d0, Rmax, tandip 
        
        Args:
            hists: Dictionary of histograms
            out_path: Output path for saving figure
            toggle_lines (dict): Dictionary to control which threshold lines to show.
                               Defaults to all True if not provided.
                               Keys: "mom_full", "pz", "t0", "trkqual", "nactive", 
                                     "t0err", "d0", "maxr", "pitch_angle"
        """
        # 3x3 subplots
        fig, ax = plt.subplots(3, 3, figsize=(6.4*3, 4.8*3))
        selection = ["All", "Preselect", "CE-like"]
        
        # Set default toggle_lines if not provided
        if toggle_lines is None:
            toggle_lines = {
                "mom_full": True, "pz": True, "t0": True,
                "trkqual": True, "nactive": True, "t0err": True,
                "d0": True, "maxr": True, "pitch_angle": True
            }
        
        # Helper function to format axes
        def format_axis(ax, labels): 
            ax.set_yscale("log")
            ax.legend(labels, frameon=True, loc="upper right")
        
        # Row 0: mom, pz, t0
        # mom (momentum magnitude)
        name = "mom_full"
        h1o_mom = hists[name]
        h1o_mom = h1o_mom[{"selection": selection}]
        h1o_mom.plot1d(overlay="selection", ax=ax[0,0], yerr=False)
        labels = list(h1o_mom.axes["selection"])
        format_axis(ax[0,0], labels)
        # Add momentum window lines
        if False: # toggle_lines.get("mom_full", True):
            ax[0,0].axvline(self.analyse.thresholds["lo_ext_win_mevc"], linestyle="--", color="grey")
            ax[0,0].axvline(self.analyse.thresholds["hi_ext_win_mevc"], linestyle="--", color="grey")
            ax[0,0].axvline(self.analyse.thresholds["lo_sig_win_mevc"], linestyle="--", color="red")
            ax[0,0].axvline(self.analyse.thresholds["hi_sig_win_mevc"], linestyle="--", color="red")
        
        # pz (momentum z-component)
        name = "pz"
        h1o_pz = hists[name]
        h1o_pz = h1o_pz[{"selection": selection}]
        h1o_pz.plot1d(overlay="selection", ax=ax[0,1], yerr=False)
        labels = list(h1o_pz.axes["selection"])
        format_axis(ax[0,1], labels)
        # Add downstream line (pz > 0)
        if toggle_lines.get("pz", True):
            ax[0,1].axvline(0, linestyle="--", color="grey")
        
        # t0 (time)
        name = "t0"
        h1o_t0 = hists[name]
        h1o_t0 = h1o_t0[{"selection": selection}]
        h1o_t0.plot1d(overlay="selection", ax=ax[0,2], yerr=False)
        labels = list(h1o_t0.axes["selection"])
        format_axis(ax[0,2], labels)
        # Add t0 window lines
        if toggle_lines.get("t0", True) and self.analyse.active_cuts["within_t0"] and self.on_spill:
            ax[0,2].axvline(self.analyse.thresholds["lo_t0_ns"], linestyle="--", color="grey")
            ax[0,2].axvline(self.analyse.thresholds["hi_t0_ns"], linestyle="--", color="grey")
            
        # Row 1: trkqual, nactive, t0err
        # trkqual (track quality)
        name = "trkqual"
        h1o_trkqual = hists[name]
        h1o_trkqual = h1o_trkqual[{"selection": selection}]
        h1o_trkqual.plot1d(overlay="selection", ax=ax[1,0], yerr=False)
        labels = list(h1o_trkqual.axes["selection"])
        format_axis(ax[1,0], labels)
        # Add track quality threshold
        if toggle_lines.get("trkqual", True) and self.analyse.active_cuts["good_trkqual"]:
            ax[1,0].axvline(self.analyse.thresholds["lo_trkqual"], linestyle="--", color="grey")
        
        # nactive (number of active hits)
        name = "nactive"
        h1o_nactive = hists[name]
        h1o_nactive = h1o_nactive[{"selection": selection}]
        h1o_nactive.plot1d(overlay="selection", ax=ax[1,1], yerr=False)
        labels = list(h1o_nactive.axes["selection"])
        format_axis(ax[1,1], labels)
        # Add minimum hits threshold
        if toggle_lines.get("nactive", True) and self.analyse.active_cuts["has_hits"]:
            ax[1,1].axvline(self.analyse.thresholds["lo_nactive"], linestyle="--", color="grey")
        
        # t0err (time uncertainty)
        name = "t0err"
        h1o_t0err = hists[name]
        h1o_t0err = h1o_t0err[{"selection": selection}]
        h1o_t0err.plot1d(overlay="selection", ax=ax[1,2], yerr=False)
        labels = list(h1o_t0err.axes["selection"])
        format_axis(ax[1,2], labels)
        # Add t0err threshold
        if toggle_lines.get("t0err", True) and self.analyse.active_cuts["within_t0err"]:
            ax[1,2].axvline(self.analyse.thresholds["hi_t0err"], linestyle="--", color="grey")
        
        # Row 2: d0, Rmax, tandip
        # d0 (distance of closest approach)
        name = "d0"
        h1o_d0 = hists[name]
        h1o_d0 = h1o_d0[{"selection": selection}]
        h1o_d0.plot1d(overlay="selection", ax=ax[2,0], yerr=False)
        labels = list(h1o_d0.axes["selection"])
        format_axis(ax[2,0], labels)
        # Add d0 threshold
        if toggle_lines.get("d0", True) and self.analyse.active_cuts["within_d0"]:
            ax[2,0].axvline(self.analyse.thresholds["hi_d0_mm"], linestyle="--", color="grey")
        
        # maxr (maximum radius)
        name = "maxr"
        h1o_maxr = hists[name]
        h1o_maxr = h1o_maxr[{"selection": selection}]
        h1o_maxr.plot1d(overlay="selection", ax=ax[2,1], yerr=False)
        labels = list(h1o_maxr.axes["selection"])
        format_axis(ax[2,1], labels)
        # Add maxr thresholds
        if toggle_lines.get("maxr", True):
            if self.analyse.active_cuts["within_lhr_max_lo"]:
                ax[2,1].axvline(self.analyse.thresholds["lo_maxr_mm"], linestyle="--", color="grey")
            if self.analyse.active_cuts["within_lhr_max_hi"]:
                ax[2,1].axvline(self.analyse.thresholds["hi_maxr_mm"], linestyle="--", color="grey")
            
        # pitch_angle (tan dip)
        name = "pitch_angle"
        h1o_pitch_angle = hists[name]
        h1o_pitch_angle = h1o_pitch_angle[{"selection": selection}]
        h1o_pitch_angle.plot1d(overlay="selection", ax=ax[2,2], yerr=False)
        labels = list(h1o_pitch_angle.axes["selection"])
        format_axis(ax[2,2], labels)
        # Add tan dip thresholds
        if toggle_lines.get("pitch_angle", True):
            if self.analyse.active_cuts["within_pitch_angle_lo"]:
                ax[2,2].axvline(self.analyse.thresholds["lo_pitch_angle"], linestyle="--", color="grey")
            if self.analyse.active_cuts["within_pitch_angle_hi"]:
                ax[2,2].axvline(self.analyse.thresholds["hi_pitch_angle"], linestyle="--", color="grey")
                
        # if toggle_lines.get("pitch_angle", True) and self.analyse.active_cuts["within_pitch_angle"]:
        #     ax[2,2].axvline(self.analyse.thresholds["lo_tanDip"], linestyle="--", color="grey")
        #     ax[2,2].axvline(self.analyse.thresholds["hi_tanDip"], linestyle="--", color="grey")
        
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=300)
            self.logger.log(f"\tWrote {out_path}", "success")
        plt.show()

    # Old ratio plot 
    # Keeping it here because it was painful to make
    # def z_plots(hists, out_path=f"../../img/{ds_type}/h1_ana_crv_z_{ds_type}_{tag}.png"):
    #     fig, (ax_main, ax_ratio) = plt.subplots(
    #         2, 1, figsize=(6.4*1.25, 4.8*1.25), 
    #         gridspec_kw={'height_ratios': [2, 1], 'hspace': 0},
    #         sharex=True  # This ensures alignment
    #     )
        
    #     h_wide = hists['CRV z-position']
    #     h_wide = h_wide[{"selection": ["All", "CE-like"]}]
        
    #     # Main plot
    #     h_wide.plot1d(overlay='selection', ax=ax_main, density=True)
    #     ax_main.set_ylabel("Norm. coincidences")
    #     # ax_main.set_yscale("log")
    #     ax_main.grid(True, alpha=0.3)
    #     ax_main.legend(frameon=False, loc="best")
    #     ax_main.set_xlabel("")  # Remove x-label from main plot
    #     ax_main.set_xticklabels([])
    #     ax_main.set_title("Off-spill")
    
    #     ax_main.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True)
        
    #     # Ratio plot
    #     h_All = h_wide[{"selection": "All"}] 
    #     h_ce = h_wide[{"selection": "CE-like"}]
    
    #     h_All = h_All / h_All.sum()
    #     h_ce = h_ce / h_ce.sum()
        
    #     # Calculate ratio (CE-like / All)
    #     ratio = h_ce.values() / h_All.values()
    #     bin_centers = h_ce.axes[0].centers
        
    #     ax_ratio.plot(bin_centers, ratio, 'ko-', markersize=3)
    #     # ax_ratio.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        
    #     ax_ratio.set_ylabel("Ratio", va='center',ha='center')
    #     ax_ratio.set_xlabel("z-position [mm]")
    #     ax_ratio.grid(True, alpha=0.3)
    #     # ax_ratio.set_ylim(0, 2) 
    
    #     # Align y-axis labels horizontAlly
    #     # ax_main.yaxis.set_label_coords(-0.1, 0.5)
    #     ax_ratio.yaxis.set_label_coords(-0.1, 0.5)
    
    #     plt.tight_layout()
    #     plt.savefig(out_path, dpi=300)
    #     plt.show()
    
    #     logger.log(f"Wrote {out_path}", "success")
    
    #     # Separate CE-like only plot
    #     fig_ce, ax_ce = plt.subplots() # figsize=(6.4*1.25, 4.8*1.25))
        
    #     h_ce_only = hists['CRV z-position'][{"selection": "CE-like"}]
    #     h_ce_only.plot1d(ax=ax_ce, density=False, color='blue')  # Or whatever color you prefer
        
    #     ax_ce.set_ylabel("Coincidences")
    #     ax_ce.set_xlabel("z-position [mm]")
    #     ax_ce.set_title("CE-like tracks (off-spill)")
    #     ax_ce.grid(True, alpha=0.3)
    #     ax_ce.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True)
        
    #     plt.tight_layout()
        
    #     # Save CE-like plot with modified filename
    #     out_path_ce = out_path.replace(".png", "_ce_only.png")
    #     plt.savefig(out_path_ce, dpi=300)
    #     plt.show()
    
    #     logger.log(f"Wrote {out_path_ce}", "success")
        