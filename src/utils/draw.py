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
    
    # Style constants 

    # # v1 - step histograms
    # STEP_STYLES = {
    #     "All": {
    #         "color": "#2E74B5",      # Professional blue
    #         "linewidth": 2.0,
    #         "alpha": 1.0,
    #         "linestyle": "-"
    #     },
    #     "Preselect": {
    #         "color": "#CC8400",      # Amber/gold
    #         "linewidth": 2.0,
    #         "alpha": 1.0,
    #         "linestyle": "-"
    #     },
    #     "CE-like": {
    #         "color": "#228B22",      # Forest green (signal)
    #         "linewidth": 2.0,        # Thicker for importance
    #         "alpha": 1.0,
    #         "linestyle": "-"
    #     },
    #     "Unvetoed": {
    #         "color": "#C41E3A",      # Cardinal red
    #         "linewidth": 2.0,
    #         "alpha": 1.0,
    #         "linestyle": "-"
    #     }
    # }

    # # v2
    # STEP_STYLES = {
    #     "All": {
    #         "color": "#C41E3A",  
    #         "linewidth": 1.5,
    #         "alpha": 0.4,
    #         "linestyle": ":"
    #     },
    #     "Preselect": {
    #         "color": "#C41E3A",  
    #         "linewidth": 2.0,
    #         "alpha": 0.6,
    #         "linestyle": "--"
    #     },
    #     "CE-like": {
    #         "color": "#C41E3A",  
    #         "linewidth": 2.5,
    #         "alpha": 0.8,
    #         "linestyle": "-"
    #     },
    #     "Unvetoed": {
    #         "color": "#C41E3A", 
    #         "linewidth": 3.0,
    #         "alpha": 1.0,
    #         "linestyle": "-"
    #     }
    # }

    # v3


    def __init__(self, cutset_name="alpha", on_spill=False, hist_styles=None, line_styles=None, colourblind=True, verbosity=1): 
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
            "CE-like": {
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
            "CE-like": {
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

            
    
            
        # h_sel = hist_obj[{"selection": selection}]
        # h_sel.plot1d(
        #     overlay="selection", 
        #     ax=ax, 
        #     histtype=histtype, 
        #     yerr=False,
        #     density=density,
        #     color=colors,
        #     flow="none",
        #     linewidth=linewidths[0] if len(set(linewidths)) == 1 else linewidths,
        #     alpha=alphas[0] if len(set(alphas)) == 1 else alphas
        # )
        
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
                         title="Wide range: 0-1000 MeV/c")
    
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
            title=title,
            ncols=2,
            y_ext_factor=40,
            loc="upper center"
        ) 
        
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
            plt.savefig(out_path, dpi=300)
            self.logger.log(f"\tWrote {out_path}", "success")
        plt.show()

    def plot_crv_z_long(self, hists, out_path=None):
        """Plot CRV z-position histograms"""
        selection = ["All", "Preselect", "CE-like"]
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
        """Plot 3x3 summary plot"""
        fig, ax = plt.subplots(3, 3, figsize=(3*6.4, 3*4.8))
        fig.subplots_adjust(hspace=0.3, wspace=0.25)
        
        # selection = ["All", "Preselect", "CE-like", "Unvetoed"]
        
        # Set default toggle_lines if not provided
        if toggle_lines is None:
            toggle_lines = {
                "mom_full": True, "pz": True, "t0": True,
                "trkqual": True, "nactive": True, "t0err": True,
                "d0": True, "maxr": True, "pitch_angle": True
            }
        
        # Variable info for axis labels
        var_info = {
            "mom_full": ("Momentum [MeV/c]", "", "upper right", 1.0, 1), # xlabel, title, loc, y_ext_factor, ncols
            "mom_z": (r"$p_{z}$ [MeV/c]", "", "upper right", 20, 2), 
            "t0": ("Track fit time [ns]", "", "upper center", 20, 2), 
            "trkqual": ("Track quality", "", "upper right", 5, 2), 
            "nactive": ("Active tracker hits", "", "upper right", 20, 2), 
            "t0err": (r"$\sigma_{t_{0}}$ [ns]", "", "upper right", 1, 2), 
            "d0": (r"$d_{0}$ [mm]", "", "upper right", 20, 2), 
            "maxr": (r"$R_{\text{max}}$ [mm]", "", "upper left", 12, 1), 
            "pitch_angle": (r"$p_{z}/p_{T}$", "", "upper right", 20, 1), 
        }
        
        # Plot all 9 histograms
        plot_positions = [
            ("mom_full", (0, 0)), ("mom_z", (0, 1)), ("t0", (0, 2)),
            ("trkqual", (1, 0)), ("nactive", (1, 1)), ("t0err", (1, 2)),
            ("d0", (2, 0)), ("maxr", (2, 1)), ("pitch_angle", (2, 2))
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
            "mom_res": (r"$\delta p = p_{\text{reco}} - p_{\text{truth}}$ [MeV/c]", "", "upper left", 15, 2)
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
            ax.axvline(self.analyse.thresholds["lo_nactive"], **line_kwargs) #  - 0.5
            
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