import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    def __init__(self, cutset_name="SU2020b", on_spill=False, hist_styles=None, line_styles=None, colourblind=True, verbosity=1): 
        """
        Initialise 
        
        Args:
            cutset_name (str, opt): The cutset name. Defaults to "SU2020b".
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

        self.hist_styles = {
            "All": {
                "color": "#121212", # dark grey
                "linewidth": 1.5,
                "alpha": 0.8,
                "linestyle": ":",
                "histtype" : "step"
            },
            "Preselect": {
                "color": "#121212", # dark grey
                "linewidth": 1.5,
                "alpha": 0.8,
                "linestyle": "-",
                "histtype" : "step"
            },
            "Select": {
                "color": "#228B22", # forest green
                "linewidth": 1.5,
                "alpha": 0.5,
                "linestyle": "-",
                "histtype" : "bar"
            },
            "Unvetoed": {
                "color": "#C41E3A", # cardinal red
                "linewidth": 1.5,
                "alpha": 0.8,
                "linestyle": "-",
                "histtype" : "bar"
            },
            # ML stuff
            "Before cuts": {
                "color": "#228B22", # forest green
                "linewidth": 1.5,
                "alpha": 0.5,
                "linestyle": "-",
                "histtype" : "bar"
            },
            "After cuts": {
                "color": "#C41E3A", # cardinal red
                "linewidth": 1.5,
                "alpha": 0.8,
                "linestyle": "-",
                "histtype" : "bar"
            }
        } if hist_styles is None else hist_styles

        colourblind_styles = {
            "All": {
                "color": "#000000",  # black
                "linewidth": 1.0,
                "alpha": 0.7,
                "linestyle": ":",
                "histtype": "step"
            },
            "Preselect": {
                "color": "#000000",  # black  
                "linewidth": 2.0,
                "alpha": 0.9,
                "linestyle": "-",
                "histtype": "step"
            },
            "Select": {
                "color": "#0173B2",  # blue (colorblind safe)
                "linewidth": 2.0,
                "alpha": 0.6,
                "linestyle": "-",
                "histtype": "bar"
            },
            "Unvetoed": {
                "color": "#FF6600",  # bright orange 
                "linewidth": 2.0,
                "alpha": 0.8,
                "linestyle": "-", 
                "histtype": "bar"
            },
            # ML stuff
            "Before cuts": {
                "color": "#121212",  # dark grey 
                "linewidth": 2.0,
                "alpha": 0.6,
                "linestyle": "-",
                "histtype": "step"
            },
            "After cuts": {
                "color": "#0173B2",  # blue 
                "linewidth": 2.0,
                "alpha": 0.6,
                "linestyle": "-", 
                "histtype": "bar"
            }
        }

        self.hist_styles = colourblind_styles

        # Threshold line styling
        self.line_styles = { 
            "linestyle": "--", 
            "color": "black", 
            "linewidth": 1.5, 
            "alpha": 0.8
        } if line_styles is None else line_styles

        # Set high DPI globally for all plots
        # plt.rcParams['figure.dpi'] = 300
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"
        
        # Get colors from style file cycle (fallback)
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        self.colours = [color["color"] for color in prop_cycle]
        
        # Confirm
        self.logger.log(f"Initialised", "info")

    def _count_events(self, hists, selection, label):
        """Utility to count events from hist selections"""
        h = hists[selection] 
        h = h[{"selection": label}]  
        return int(h.sum())

    def _format_axis(self, ax, labels, xlabel="", ylabel="", title="", y_ext_factor=1, ncols=1, leg=True, log=True, frameon=False, loc="upper right"):
        """Apply standard axis formatting"""
        if log:
            ax.set_yscale("log")
        if leg:
            ax.legend(labels, frameon=frameon, ncols=ncols, loc=loc)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        # Extend y-axis by multiplicative factor
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], current_ylim[1] * y_ext_factor)  # 50% more space

    def _plot_histogram(self, hist_obj, ax, selection, density=False):
        """Plot clean step histograms"""
        # Get selection
        h_sel = hist_obj[{"selection": selection}]
        
        # Get the selection labels
        labels = list(h_sel.axes["selection"])
        
        for label in labels:
            if label in self.hist_styles:
                style = self.hist_styles[label]
                color = style["color"] if style["color"] is not None else "red"
                linewidth = style["linewidth"] if style["linewidth"] is not None else 1.5
                linestyle = style["linestyle"] if style["linestyle"] is not None else "-"
                alpha = style["alpha"] if style["alpha"] is not None else 0.9
                histtype = style["histtype"] if style["histtype"] is not None else "step"
            else:
                # Fallback styling
                color = "red"  
                linewidth = 1.5
                linestyle = "-"
                alpha = 0.9  
                histtype = "step"
        
            # Plot histograms ONE-BY-ONE 
            hist_obj[{"selection": label}].plot1d(
                ax=ax, 
                yerr=False,
                density=density,
                color=color,
                edgecolor=color,
                flow="none",
                histtype=histtype,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle  
            )

        return labels
    
    def plot_mom_windows(self, hists, out_path=None):
        """Plot 2x2 momentum window histograms"""
        fig, ax = plt.subplots(2, 2, figsize=(2*6.4, 2*4.8))
        
        # Full range 
        name = "mom_full"
        h1o_mom_wide = hists[name]
        labels = self._plot_histogram(h1o_mom_wide, ax[0,0], 
                                     list(h1o_mom_wide.axes["selection"]))
        
        # Add event counts to labels
        for i, label in enumerate(labels):
            labels[i] = f"{label}: {self._count_events(hists, name, label):,}"
        
        self._format_axis(ax[0,0], labels, 
                         xlabel="Momentum [MeV/c]",
                         ylabel="Tracks",
                         title="Inclusive window: 0-1000 MeV/c")

        # Wide window 
        name = "mom_wide"
        h1o_mom_ext = hists[name]
        labels = self._plot_histogram(h1o_mom_ext, ax[0,1],
                                     list(h1o_mom_ext.axes["selection"]))
        
        for i, label in enumerate(labels):
            labels[i] = f"{label}: {self._count_events(hists, name, label):,}"
        
        title = f"Wide window: {self.analyse.thresholds['lo_wide_win_mevc']}" \
                f"-{self.analyse.thresholds['hi_wide_win_mevc']} MeV/c"
        self._format_axis(ax[0,1], labels, 
            xlabel="Momentum [MeV/c]",
            ylabel="Tracks",
            title=title,
            ncols=2,
            y_ext_factor=40,
            loc="upper center"
        ) 
        
        # Extended window 
        name = "mom_ext"
        h1o_mom_ext = hists[name]
        labels = self._plot_histogram(h1o_mom_ext, ax[1,0],
                                     list(h1o_mom_ext.axes["selection"]))
        
        for i, label in enumerate(labels):
            labels[i] = f"{label}: {self._count_events(hists, name, label):,}"
        
        title = f"Extended window: {self.analyse.thresholds['lo_ext_win_mevc']}" \
                f"-{self.analyse.thresholds['hi_ext_win_mevc']} MeV/c"
        self._format_axis(ax[1,0], labels, 
            xlabel="Momentum [MeV/c]",
            ylabel="Tracks",
            title=title,
            ncols=2,
            y_ext_factor=40,
            loc="upper center"
        ) 
        
        # Signal window 
        name = "mom_sig"
        h1o_mom_sig = hists[name]
        labels = self._plot_histogram(h1o_mom_sig, ax[1,1],
                                     list(h1o_mom_sig.axes["selection"]))
        
        for i, label in enumerate(labels):
            labels[i] = f"{label}: {self._count_events(hists, name, label):,}"
        
        title = f"Signal window: {self.analyse.thresholds['lo_sig_win_mevc']}" \
                f"-{self.analyse.thresholds['hi_sig_win_mevc']} MeV/c"

        self._format_axis(ax[1,1], labels, 
            xlabel="Momentum [MeV/c]",
            ylabel="Tracks",
            title=title,
            ncols=2,
            y_ext_factor=40,
            loc="upper center"
        ) 
        
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=300)
            self.logger.log(f"\tWrote {out_path}", "success")
        plt.show()
        
    def plot_crv_z(self, hists, out_path=None):
        """Plot CRV z-position histograms"""
        selection = ["All", "Preselect", "Select"]
        fig, ax = plt.subplots(figsize=(8, 6)) 
        
        name = "crv_z"
        h1o_z = hists[name]
        h1o_z = h1o_z[{"selection": selection}]
        
        # Plot step histograms
        self._plot_histogram(h1o_z, ax, selection, density=True)
        
        # Format axis
        self._format_axis(ax, selection, 
                         xlabel="z-position [mm]", 
                         ylabel="Normalised coincidences",
                         log=False,
                         frameon=False)
        
        # Scientific notation
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
    
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=300)
            self.logger.log(f"\tWrote {out_path}", "success")
        plt.show()

    def plot_crv_z_long(self, hists, out_path=None):
        """Plot CRV z-position histograms"""
        selection = ["All", "Preselect", "Select"]
        fig, ax = plt.subplots(figsize=(8*2, 6/1.5)) 
        
        name = "crv_z"
        h1o_z = hists[name]
        h1o_z = h1o_z[{"selection": selection}]
        
        # Plot step histograms
        self._plot_histogram(h1o_z, ax, selection, density=True)
        
        # Format axis
        self._format_axis(ax, selection, 
                         xlabel="z-position [mm]", 
                         ylabel="Normalised coincidences",
                         log=False,
                         frameon=False)
        
        # Scientific notation
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
    
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path)
            self.logger.log(f"\tWrote {out_path}", "success")
        plt.show()

    def plot_summary(self, hists, out_path=None, toggle_lines=None):
        """Plot 4x3 summary plot"""
        fig, ax = plt.subplots(4, 3, figsize=(3*6.4, 4*4.8))
        fig.subplots_adjust(hspace=0.3, wspace=0.25)
        
        # selection = ["All", "Preselect", "Select", "Unvetoed"]
        
        # Set default toggle_lines if not provided
        if toggle_lines is None:
            toggle_lines = {
                "pz": True, "t0": True, "trkqual": True,
                "nactive": True, "t0err": True, "d0": True, 
                "maxr": True, "pitch_angle": True, "dT": True
            }
        
        # Variable info for axis labels
        var_info = {
            # 
            "mom_full": (r"Momentum [MeV/c]", "", "upper right", 20, 2), 
            "mom_z": (r"$p_{z}$ [MeV/c]", "", "upper right", 20, 2), 
            "t0": ("Track fit time [ns]", "", "upper center", 20, 2), 
            # 
            "trkqual": ("Track quality", "", "upper right", 5, 2), 
            "nactive": ("Active tracker hits", "", "upper right", 20, 2), 
            "t0err": (r"$\sigma_{t_{0}}$ [ns]", "", "upper right", 1, 2), 
            #
            "d0": (r"$d_{0}$ [mm]", "", "upper right", 20, 2), 
            "maxr": (r"$R_{\text{max}}$ [mm]", "", "upper left", 12, 1), 
            "pitch_angle": (r"$p_{z}/p_{T}$", "", "upper right", 20, 1), 
            #
            "dT": (r"$\Delta t$ [ns]", "", "upper left", 5, 1), 
            "coinc_start_time": (r"Start time [ns]", "", "upper right", 5, 2), 
            "coinc_end_time": (r"End time [ns]", "", "upper left", 5, 2),
        }

        plot_positions = [
            ("mom_full", (0, 0)), ("mom_z", (0, 1)), ("t0", (0, 2)),
            ("trkqual", (1, 0)), ("nactive", (1, 1)), ("t0err", (1, 2)),
            ("d0", (2, 0)), ("maxr", (2, 1)), ("pitch_angle", (2, 2)),
            ("dT", (3, 0)), ("coinc_start_time", (3, 1)), ("coinc_end_time", (3, 2))
        ]
        
        for var_name, (row, col) in plot_positions:
            # Plot histograms
            labels = self._plot_histogram(hists[var_name], ax[row, col], list(hists[var_name].axes["selection"])) # , selection)
            
            # Get axis labels and title
            xlabel, title, loc, y_ext_factor, ncols = var_info[var_name]
            ylabel = "Tracks" if col==0 else ""
            
            # Only show legend on first subplot
            show_legend = True
            
            # Apply formatting
            self._format_axis(
                ax[row, col], 
                labels, 
                xlabel=xlabel, 
                ylabel=ylabel, 
                title=title, 
                leg=show_legend,
                loc=loc,
                y_ext_factor=y_ext_factor,
                ncols=ncols
                
            )
            
            # Add threshold lines
            self._add_threshold_lines(ax[row, col], var_name, toggle_lines, self.line_styles)
        
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path)
            self.logger.log(f"\tWrote {out_path}", "success")
        plt.show()
        
    def plot_mom_summary(self, hists, out_path=None):
        
        fig, ax = plt.subplots(2, 2, figsize=(6.4*2, 2*4.8))        # Variable info for axis labels
        var_info = {
            
            "mom_z": (r"$p_{z}$ [MeV/c]", "", "upper right", 20, 2), # xlabel, title, loc, y_ext_factor, ncols
            "mom_T": (r"$p_{T}$ [MeV/c]", "", "upper right", 20, 2), 
            "mom_err": (r"$\sigma_{p}$ [MeV/c]", "", "upper right", 1.0, 1), 
            "mom_res": (r"$\delta p = p_{\text{reco}} - p_{\text{truth}}$ [MeV/c]", "", "upper left", 10, 1)
        }
                 
        # Plot all 4 histograms
        plot_positions = [
            ("mom_z", (0, 0)), ("mom_T", (0, 1)), 
            ("mom_err", (1, 0)), ("mom_res", (1, 1))
        ]        
        for var_name, (row, col) in plot_positions:
            # Plot histograms
            labels = self._plot_histogram(hists[var_name], ax[row, col], list(hists[var_name].axes["selection"])) # , selection)
            
            # Get axis labels and title
            xlabel, title, loc, y_ext_factor, ncols = var_info[var_name]
            ylabel = "Tracks" if col==0 else ""
            
            # Only show legend on first subplot
            show_legend = True # (row == 0 and col == 0)
            
            # Apply formatting
            self._format_axis(
                ax[row, col], 
                labels, 
                xlabel=xlabel, 
                ylabel=ylabel, 
                title=title, 
                leg=show_legend,
                loc=loc,
                y_ext_factor=y_ext_factor,
                ncols=ncols
                
            )
        
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path)
            self.logger.log(f"\tWrote {out_path}", "success")
        plt.show()
        
    def plot_ml_summary(self, hists, out_path=None, toggle_lines=None):
        """Plot 4x3 ML summary plot with all parameters (original ML + track quality)"""
        
        fig, ax = plt.subplots(4, 3, figsize=(3*6.4, 4*4.8))
        fig.subplots_adjust(hspace=0.3, wspace=0.25)
        
        # Set default toggle_lines if not provided
        if toggle_lines is None:
            toggle_lines = {
                "mom_full": False, "mom_z": True, "pdg": False,
                "trk_per_event": False, "t0": True, "trkqual": True,
                "nactive": True, "t0err": True, "d0": True, 
                "maxr": True, "pitch_angle": True, "dT": True
            }
        
        # Variable info for axis labels
        var_info = {
            # Original ML parameters
            "mom_full": (r"Momentum [MeV/c]", "", "upper right", 1.0, 1), # xlabel, title, loc, y_ext_factor, ncols
            "mom_z": (r"$p_{z}$ [MeV/c]", "", "upper right", 20, 2), 
            "pdg": (r"PDG", "", "upper center", 1.0, 1), 
            "trk_per_event": (r"Tracks per event", "", "upper right", 1.0, 1),
            # Track quality parameters
            "t0": ("Track fit time [ns]", "", "upper center", 20, 2), 
            "trkqual": ("Track quality", "", "upper right", 5, 2), 
            "nactive": ("Active tracker hits", "", "upper right", 20, 2), 
            "t0err": (r"$\sigma_{t_{0}}$ [ns]", "", "upper right", 1, 2), 
            "d0": (r"$d_{0}$ [mm]", "", "upper right", 20, 2), 
            "maxr": (r"$R_{\text{max}}$ [mm]", "", "upper left", 12, 1), 
            "pitch_angle": (r"$p_{z}/p_{T}$", "", "upper right", 20, 1), 
            "dT": (r"$\Delta t$ [ns]", "", "upper left", 5, 1), 
        }
        
        # Plot all 12 histograms (4 rows x 3 columns)
        plot_positions = [
            # Row 0: Original ML parameters
            ("mom_full", (0, 0)), ("mom_z", (0, 1)), ("pdg", (0, 2)),
            # Row 1: ML + track quality
            ("trk_per_event", (1, 0)), ("t0", (1, 1)), ("trkqual", (1, 2)),
            # Row 2: Track quality parameters
            ("nactive", (2, 0)), ("t0err", (2, 1)), ("d0", (2, 2)),
            # Row 3: Additional track quality
            ("maxr", (3, 0)), ("pitch_angle", (3, 1)), ("dT", (3, 2))
        ]
        
        for var_name, (row, col) in plot_positions:
            # Plot histograms
            labels = self._plot_histogram(hists[var_name], ax[row, col], list(hists[var_name].axes["selection"]))
            
            # Get axis labels and title
            xlabel, title, loc, y_ext_factor, ncols = var_info[var_name]
            ylabel = "Tracks" if col==0 else ""
            
            # Show legend on all subplots
            show_legend = True
            
            # Apply formatting
            self._format_axis(
                ax[row, col], 
                labels, 
                xlabel=xlabel, 
                ylabel=ylabel, 
                title=title, 
                leg=show_legend,
                loc=loc,
                y_ext_factor=y_ext_factor,
                ncols=ncols
            )
            
            # Add threshold lines
            self._add_threshold_lines(ax[row, col], var_name, toggle_lines, self.line_styles)
        
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path)
            self.logger.log(f"\tWrote {out_path}", "success")
        plt.show()
        
    def _add_threshold_lines(self, ax, var_name, toggle_lines, line_kwargs):
        """Add threshold lines to plots"""
        if not toggle_lines.get(var_name, True):
            return
            
        # Add threshold lines based on variable
        if var_name == "pz":
            ax.axvline(0, **line_kwargs)
            
        elif var_name == "t0" and self.analyse.active_cuts["within_t0"] and self.on_spill:
            ax.axvline(self.analyse.thresholds["lo_t0_ns"], **line_kwargs)
            ax.axvline(self.analyse.thresholds["hi_t0_ns"], **line_kwargs)
            
        elif var_name == "trkqual" and self.analyse.active_cuts["good_trkqual"]:
            ax.axvline(self.analyse.thresholds["lo_trkqual"], **line_kwargs)
            
        elif var_name == "nactive" and self.analyse.active_cuts["has_hits"]:
            ax.axvline(self.analyse.thresholds["lo_nactive"], **line_kwargs) 
            
        elif var_name == "t0err" and self.analyse.active_cuts["within_t0err"]:
            ax.axvline(self.analyse.thresholds["hi_t0err"], **line_kwargs)
            
        elif var_name == "d0" and self.analyse.active_cuts["within_d0"]:
            ax.axvline(self.analyse.thresholds["hi_d0_mm"], **line_kwargs)
            
        elif var_name == "maxr":
            if self.analyse.active_cuts["within_lhr_max_lo"]:
                ax.axvline(self.analyse.thresholds["lo_maxr_mm"], **line_kwargs)
            if self.analyse.active_cuts["within_lhr_max_hi"]:
                ax.axvline(self.analyse.thresholds["hi_maxr_mm"], **line_kwargs)
                
        elif var_name == "pitch_angle":
            if self.analyse.active_cuts["within_pitch_angle_lo"]:
                ax.axvline(self.analyse.thresholds["lo_pitch_angle"], **line_kwargs)
            if self.analyse.active_cuts["within_pitch_angle_hi"]:
                ax.axvline(self.analyse.thresholds["hi_pitch_angle"], **line_kwargs)

        elif var_name == "dT":
            if self.analyse.active_cuts["unvetoed"]:
                ax.axvline(self.analyse.thresholds["lo_veto_dt_ns"], **line_kwargs)
                ax.axvline(self.analyse.thresholds["hi_veto_dt_ns"], **line_kwargs)

        elif var_name == "coinc_start_time":
            if self.analyse.active_cuts["within_coinc_start_time"]:
                ax.axvline(self.analyse.thresholds["lo_crv_start_ns"], **line_kwargs)

        elif var_name == "coinc_end_time":
            if self.analyse.active_cuts["within_coinc_end_time"]:
                ax.axvline(self.analyse.thresholds["hi_crv_end_ns"], **line_kwargs)
                
        elif var_name == "dT":
            if self.analyse.active_cuts["unvetoed"]:
                ax.axvline(self.analyse.thresholds["lo_veto_dt_ns"], **line_kwargs)
                ax.axvline(self.analyse.thresholds["hi_veto_dt_ns"], **line_kwargs)

    def draw_cosmic_parents_from_array(self, events, out_path=None, percentage=False, p_threshold=None):
        """
        Plot cosmic parent PDG codes directly from events array (simpler approach)
        
        Args:
            events: Events array containing trkmc information
            out_path: Path to save figure
            percentage: If True, show percentages instead of counts
            p_threshold: Momentum threshold label (for plot title)
        """
        import numpy as np
        import awkward as ak
        
        # PDG to particle name mapping
        pdg_to_label = {
            11: r"$e^{-}$", -11: r"$e^{+}$",
            13: r"$\mu^{-}$", -13: r"$\mu^{+}$", 
            2112: "n", -2112: r"$\bar{n}$",
            2212: "p", -2212: r"$\bar{p}$",
            22: r"$\gamma$", 111: r"$\pi^{0}$",
            211: r"$\pi^{+}$", -211: r"$\pi^{-}$"
        }
        
        # Extract cosmic parent PDG codes from events
        try:
            trkmc = events["trkmc"]
            trkmcsim = trkmc["trkmcsim"]
            
            # Find cosmic parents (rank == -1)
            rank_mask = trkmcsim["rank"] == -1
            
            # For each track, get the highest momentum cosmic parent
            mom_x = trkmcsim["mom"]["fCoordinates"]["fX"]
            mom_y = trkmcsim["mom"]["fCoordinates"]["fY"]
            mom_z = trkmcsim["mom"]["fCoordinates"]["fZ"]
            mom_mag = np.sqrt(mom_x**2 + mom_y**2 + mom_z**2)
            
            # Apply rank mask first
            cosmic_parents = trkmcsim[rank_mask]
            cosmic_mom_mag = mom_mag[rank_mask]
            
            # Find max momentum cosmic parent per track
            max_mom_mask = cosmic_mom_mag == ak.max(cosmic_mom_mag, axis=-1)
            
            # Extract PDG codes
            cosmic_parent_pdg = cosmic_parents["pdg"][max_mom_mask]
            
            # Flatten to 1D array
            pdg_flat = ak.flatten(cosmic_parent_pdg, axis=None)
            
            # Count occurrences
            unique_pdg, counts = np.unique(ak.to_numpy(pdg_flat), return_counts=True)
            
        except Exception as e:
            self.logger.log(f"Error extracting cosmic parents: {e}", "error")
            return
        
        if len(unique_pdg) == 0:
            self.logger.log("No cosmic parent data found in events array", "warning")
            return
        
        # Sort by count (descending)
        sorted_indices = np.argsort(counts)[::-1]
        pdg_codes = unique_pdg[sorted_indices]
        counts = counts[sorted_indices]
        
        # Convert PDG codes to labels
        labels = [pdg_to_label.get(int(pdg), f"PDG={int(pdg)}") for pdg in pdg_codes]
        
        # Convert to percentage if requested
        if percentage:
            counts = (counts / np.sum(counts)) * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate bar width based on number of bars
        n_bars = len(pdg_codes)
        bar_width = min(0.8, 3.0 / n_bars)
        if n_bars == 2:
            bar_width = 0.5
        elif n_bars == 3:
            bar_width = 0.67
        
        # Plot bars
        indices = np.arange(len(labels))
        bars = ax.bar(
            indices, counts, 
            width=bar_width,
            color="#C41E3A",
            edgecolor="#C41E3A",
            alpha=0.8,
            linewidth=1
        )
        
        # Set labels and formatting
        ax.set_xticks(indices)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel("Cosmic parent")
        
        ylabel = "Percentage of tracks (%)" if percentage else "Tracks"
        ax.set_ylabel(ylabel)
        
        # Add title showing momentum threshold
        title = "Cosmic parents"
        if p_threshold:
            title += f" (p > {p_threshold} MeV/c)"
        ax.set_title(title)
        
        # Format y-axis for large numbers
        if ax.get_ylim()[1] > 999:
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        if out_path:
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            self.logger.log(f"\tWrote {out_path}", "success")
        
        plt.show()
    
    # Deprecated - use draw_cosmic_parents_from_array() instead
    # def plot_cosmic_parents(self, hists, selection="Select", out_path=None, percentage=False):