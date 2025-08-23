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
    
    # Style constants - PARTICLE PHYSICS INSPIRED STYLING
    
    # Clean, professional colors with good contrast
    STEP_STYLES = {
        "All": {
            "color": "#2E74B5",      # Professional blue
            "linewidth": 2.0,
            "alpha": 1.0,
            "linestyle": "-"
        },
        "Preselect": {
            "color": "#CC8400",      # Amber/gold
            "linewidth": 2.0,
            "alpha": 1.0,
            "linestyle": "-"
        },
        "CE-like": {
            "color": "#228B22",      # Forest green (signal)
            "linewidth": 2.0,        # Thicker for importance
            "alpha": 1.0,
            "linestyle": "-"
        },
        "Unvetoed": {
            "color": "#C41E3A",      # Cardinal red
            "linewidth": 2.0,
            "alpha": 1.0,
            "linestyle": "-"
        }
    }

    def __init__(self, cutset_name="alpha", on_spill=False, verbosity=1): 
        """
        Initialise 
        
        Args:
            cutset_name (str, opt): The cutset name. Defaults to "alpha".
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

        # Get colors from style file cycle (fallback)
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        self.colours = [color["color"] for color in prop_cycle]

    def _count_events(self, hists, selection, label):
        """Utility to count events from hist selections"""
        h = hists[selection] 
        h = h[{"selection": label}]  
        return int(h.sum())

    def _format_axis(self, ax, labels, xlabel="", ylabel="", title="", leg=True, log=True, frameon=True):
        """Apply standard axis formatting"""
        if log:
            ax.set_yscale("log")
        if leg:
            ax.legend(labels, frameon=frameon)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

    def _plot_histogram(self, hist_obj, ax, selection, density=False):
        """Plot clean step histograms"""
        h_sel = hist_obj[{"selection": selection}]
        
        # Get the selection labels
        labels = list(h_sel.axes["selection"])
        
        # Prepare colors and styling
        colors = []
        linewidths = []
        linestyles = []
        alphas = []
        
        for label in labels:
            if label in self.STEP_STYLES:
                style = self.STEP_STYLES[label]
                colors.append(style["color"])
                linewidths.append(style["linewidth"])
                linestyles.append(style["linestyle"])
                alphas.append(style["alpha"])
            else:
                # Fallback styling
                colors.append(self.colours[len(colors) % len(self.colours)])
                linewidths.append(1.5)
                linestyles.append("-")
                alphas.append(0.9)
        
        # Plot step histograms
        h_sel.plot1d(
            overlay="selection", 
            ax=ax, 
            histtype="step", 
            yerr=False,
            density=density,
            color=colors,
            flow="none",
            linewidth=linewidths[0] if len(set(linewidths)) == 1 else linewidths,
            alpha=alphas[0] if len(set(alphas)) == 1 else alphas
        )
        
        return labels
    
    def plot_mom_windows(self, hists, out_path=None):
        """Plot 1x3 momentum window histograms"""
        fig, ax = plt.subplots(1, 3, figsize=(6.4*3, 4.8))
        
        # Wide range 
        name = "mom_full"
        h1o_mom_wide = hists[name]
        labels = self._plot_histogram(h1o_mom_wide, ax[0], 
                                     list(h1o_mom_wide.axes["selection"]))
        
        # Add event counts to labels
        for i, label in enumerate(labels):
            labels[i] = f"{label}: {self._count_events(hists, name, label):,}"
        
        self._format_axis(ax[0], labels, 
                         xlabel="Momentum [MeV/c]",
                         ylabel="Tracks",
                         title="Wide range: 0-100 MeV/c")
    
        # Extended window 
        name = "mom_ext"
        h1o_mom_ext = hists[name]
        labels = self._plot_histogram(h1o_mom_ext, ax[1],
                                     list(h1o_mom_ext.axes["selection"]))
        
        for i, label in enumerate(labels):
            labels[i] = f"{label}: {self._count_events(hists, name, label):,}"
        
        title = f"Extended window: {self.analyse.thresholds['lo_ext_win_mevc']}" \
                f"-{self.analyse.thresholds['hi_ext_win_mevc']} MeV/c"
        self._format_axis(ax[1], labels, 
                         xlabel="Momentum [MeV/c]",
                         title=title)
        
        # Signal window 
        name = "mom_sig"
        h1o_mom_sig = hists[name]
        labels = self._plot_histogram(h1o_mom_sig, ax[2],
                                     list(h1o_mom_sig.axes["selection"]))
        
        for i, label in enumerate(labels):
            labels[i] = f"{label}: {self._count_events(hists, name, label):,}"
        
        title = f"Signal window: {self.analyse.thresholds['lo_sig_win_mevc']}" \
                f"-{self.analyse.thresholds['hi_sig_win_mevc']} MeV/c"
        self._format_axis(ax[2], labels,
                         xlabel="Momentum [MeV/c]", 
                         title=title)
        
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path)
            self.logger.log(f"\tWrote {out_path}", "success")
        plt.show()

    def plot_crv_z(self, hists, out_path=None):
        """Plot CRV z-position histograms"""
        selection = ["All", "Preselect", "CE-like"]
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
            plt.savefig(out_path)
            self.logger.log(f"\tWrote {out_path}", "success")
        plt.show()

    def plot_summary(self, hists, out_path=None, toggle_lines=None):
        """Plot 3x3 summary plot"""
        fig, ax = plt.subplots(3, 3, figsize=(3*6.4, 3*4.8))
        fig.subplots_adjust(hspace=0.3, wspace=0.25)
        
        selection = ["All", "Preselect", "CE-like"]
        
        # Set default toggle_lines if not provided
        if toggle_lines is None:
            toggle_lines = {
                "mom_full": True, "pz": True, "t0": True,
                "trkqual": True, "nactive": True, "t0err": True,
                "d0": True, "maxr": True, "pitch_angle": True
            }
        
        # Threshold line styling
        line_kwargs = {"linestyle": "--", "color": "grey", "linewidth": 2, "alpha": 0.8}
        
        # Variable info for axis labels
        var_info = {
            "mom_full": ("Momentum [MeV/c]", ""),
            "pz": (r"$p_{z}$ [MeV/c]", ""), 
            "t0": ("Track fit time [ns]", ""),
            "trkqual": ("Track quality", ""),
            "nactive": ("Active tracker hits", ""),
            "t0err": (r"$\sigma_{t_{0}}$ [ns]", ""),
            "d0": (r"$d_{0}$ [mm]", ""),
            "maxr": (r"$R_{\text{max}}$ [mm]", ""),
            "pitch_angle": (r"$p_{z}/p_{T}$", "")
        }
        
        # Plot all 9 histograms
        plot_positions = [
            ("mom_full", (0, 0)), ("pz", (0, 1)), ("t0", (0, 2)),
            ("trkqual", (1, 0)), ("nactive", (1, 1)), ("t0err", (1, 2)),
            ("d0", (2, 0)), ("maxr", (2, 1)), ("pitch_angle", (2, 2))
        ]
        
        for var_name, (row, col) in plot_positions:
            # Plot step histograms
            labels = self._plot_histogram(hists[var_name], ax[row, col], selection)
            
            # Get axis labels and title
            xlabel, title = var_info[var_name]
            ylabel = "Tracks" if col==0 else ""
            
            # Only show legend on first subplot
            show_legend = True # (row == 0 and col == 0)
            
            # Apply formatting
            self._format_axis(ax[row, col], labels, xlabel=xlabel, ylabel=ylabel, title=title, leg=show_legend)
            
            # Add threshold lines
            self._add_threshold_lines(ax[row, col], var_name, toggle_lines, line_kwargs)
        
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
            ax.axvline(self.analyse.thresholds["lo_nactive"] - 1, **line_kwargs)
            
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