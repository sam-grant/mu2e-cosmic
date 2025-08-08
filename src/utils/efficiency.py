import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm
from pyutils.pylogger import Logger

class Efficiency():
    """
    Tools for efficiency calculations 
    """
    def __init__(self, verbosity=1):
        """Initialise
        """
        # Verbosity 
        self.verbosity = verbosity
        # Start logger
        self.logger = Logger(
            print_prefix="[Efficiency]",
            verbosity=self.verbosity
        )
        # Confirm
        self.logger.log(f"Initialised", "info")

    def _get_N_from_df(self, df_stats, row_name): 
        """
        Retrieve number of events passing from a specified row
        """
        row = df_stats[df_stats["Cut"] == row_name]
        return row["Events Passing"].iloc[0]
        
    def _get_binomial_err(self, k, N):
        """Calculate Binomial uncertainty"""
        q = k / N 
        return np.sqrt(q * (1 - q) / N)
        
    def _get_poisson_err(self, k, N):
        """Calculate Poisson uncertainty"""
        return np.sqrt(k) / N 
        
    def _get_wilson_err(self, k, N, z=1.0):
        """Calculate uncertainty with Wilson method"""
        alpha = 2 * (1 - norm.cdf(z)) # 68% confidence with z=1.0
        lower, upper = proportion_confint(k, N, alpha=alpha, method="wilson")
        return (upper - lower) / 2 # half the interval 

    def _append_result(self, results, label, k, N, eff, eff_err):
        """Add efficiency result to result list."""
        # def to_sig_figs(x, n=4):
        #     return float(f"{float(x):.{n}g}")
        result = {
            "Type": label,
            "Events Passing (k)": k,
            "Total Events (N)": N,
            "Efficiency [%]": round(float(eff * 100), 2),
            "Efficiency Error [%]": round(float(eff_err * 100), 2)
        }
        results.append(result)

    ####################################################
    # Efficiency from histograms tools 
    ####################################################

    def _get_N_from_hists(self, hists, selection, label): 
        """
        Retrieve number of events passing in a histogram
        """
        h = hists[selection] 
        h = h[{"selection": label}]  
        return int(h.sum())
        
    def get_eff_from_hists(
        self, 
        hists,
        generated_events=4e6, # typical 
        veto=True, 
        wilson=True,
        z=1.0
    ):
        """
        Report efficiency results for signal and veto analysis.
        
        Args:
            hists: Selection of histograms 
            generated_events (int, opt): Number of generated events in dataset. Defaults to 4e6.
            veto (bool, opt): Whether to include veto analysis. Defaults to True. 
            wilson (bool, opt): Use Wilson interval. Defaults to True.
            z (float, opt): Wilson interval z-score. Defaults to 1.0 (68% confidence).
            
        Returns:
            pd.DataFrame: Results containing efficiency calculations
        """
        self.logger.log(f"Getting efficiency from histogams", "info")
    
        try:
            generated_events = float(generated_events)
        except (ValueError, TypeError):
            generated_events = 0
        
        results = []
        
        def get_signal_eff(selection, title): 
            k = self._get_N_from_hists(hists, selection, label="CE-like")            
            eff = k / generated_events if int(float(generated_events)) > 0 else 0
            eff_err = self._get_binomial_err(k, generated_events)
            self._append_result(results, title, int(k), int(float(generated_events)), eff, eff_err)
    
        # Signal efficiency for CE selection over wide range
        get_signal_eff(selection="mom_full", title="Signal (wide)")
        get_signal_eff(selection="mom_ext", title="Signal (ext)")
        get_signal_eff(selection="mom_sig", title="Signal (sig)")

        def get_veto_eff(selection, title): 
            k = self._get_N_from_hists(hists, selection, label="Unvetoed")    
            N = self._get_N_from_hists(hists, selection, label="CE-like")   
            eff = (1 - k / N) if N > 0 else 0
            eff_err = self._get_wilson_err(k, N)
            self._append_result(results, title, int(k), int(N), eff, eff_err)
    
        # Signal efficiency for CE selection over wide range
        get_veto_eff(selection="mom_full", title="Veto (wide)")
        get_veto_eff(selection="mom_ext", title="Veto (ext)")
        get_veto_eff(selection="mom_sig", title="Veto (sig)")

        self.logger.log("Returning efficiency information", "success")
        
        return pd.DataFrame(results)

    # ####################################################
    # # Efficiency from cuts DataFrame 
    # ####################################################

    # def get_eff_from_df(
    #     self, 
    #     df_stats,
    #     N_gen=4e6, # typical 
    #     ce_row_name="one_reco_electron", # subject to change
    #     veto=True, 
    #     wilson=True,
    #     z=1.0
    # ):
    #     """
    #     Report efficiency results for signal and veto analysis.
        
    #     Args:
    #         df_stats (pd.DataFrame): DataFrame containing statistics
    #         generated_events (int, opt): Number of generated events in dataset. Defaults to 4e6.
    #         ce_row_name (str, opt): Name of the conversion electron row. Defaults to "one_reco_electron". 
    #         veto (bool, opt): Whether to include veto analysis. Defaults to True. 
    #         wilson (bool, opt): Use Wilson interval. Defaults to True.
    #         z (float, opt): Wilson interval z-score. Defaults to 1.0 (68% confidence).
            
    #     Returns:
    #         pd.DataFrame: Results containing efficiency calculations

    #     TODO: automatic CE-like row selection

    #     Better to get this from the histogram objects!
    #     """
    #     self.logger.log(f"Getting efficiency from DataFrame ce_row_name = {ce_row_name} and veto = {veto}", "info")
        
    #     results = []
        
    #     # Signal efficiency for CE selection over full range
    #     k_sig_full = self._get_N_from_df(df_stats, row_name=ce_row_name)
    #     eff_sig_full = k_sig_full / N_gen
    #     eff_err_sig_full = self._get_binomial_err(k_sig_full, N_gen)
    #     self._append_result_to_df(results, "Signal (full)", int(k_sig_full), int(N_gen), eff_sig_full, eff_err_sig_full)

    #     if not veto:
    #         # Signal efficiency for CE selection over extended window 
    #         k_sig_ext = self._get_N_from_df(df_stats, row_name="within_ext_win")
    #         eff_sig_ext = k_sig_ext / N_gen
    #         eff_err_sig_ext = self._get_binomial_err(k_sig_ext, N_gen)
    #         self._append_result_to_df(results, "Signal (ext)", int(k_sig_ext), int(N_gen), eff_sig_ext, eff_err_sig_ext)
    
    #         # Signal efficiency for CE selection over signal window 
    #         k_sig_sig = self._get_N_from_df(df_stats, row_name="within_sig_win")
    #         eff_sig_sig = k_sig_sig / N_gen
    #         eff_err_sig_sig = self._get_binomial_err(k_sig_sig, N_gen)
    #         self._append_result_to_df(results, "Signal (sig)", int(k_sig_sig), int(N_gen), eff_sig_sig, eff_err_sig_sig)

    #     if veto:
    #         # Veto efficiency over full range
    #         k_veto_full = self._get_N_from_df(df_stats, row_name="unvetoed")
    #         N_veto_full = self._get_N_from_df(df_stats, row_name=ce_row_name)
    #         eff_veto_full = k_veto_full / N_veto_full
    #         if wilson: 
    #             eff_err_veto_full = self._get_wilson_err(k_veto_full, N_veto_full) 
    #         if not wilson:
    #             eff_err_veto_full = self._get_poisson_err(k_veto_full, N_veto_full)
                
    #         self._append_result_to_df(results, "Veto (full)", int(k_veto_full), int(N_veto_full), eff_veto_full, eff_err_veto_full)

    #     self.logger.log("Returning efficiency information", "success")
        
    #     return pd.DataFrame(results)
    
