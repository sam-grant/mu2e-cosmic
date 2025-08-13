import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm
from scipy import stats
from pyutils.pylogger import Logger

class HistAnalyser():
    """
    Tools for histograms-based statisical analysis
    
    Efficiency and rate calculations 
    """
    def __init__(self, verbosity=1):
        """Initialise
        """
        # Verbosity 
        self.verbosity = verbosity
        # Start logger
        self.logger = Logger(
            print_prefix="[HistAnalyser]",
            verbosity=self.verbosity
        )
        # Confirm
        self.logger.log(f"Initialised", "info")

    ####################################################
    # Stats methods
    ####################################################
    
    # def _get_binomial_err(self, k, N):
    #     """Calculate Binomial uncertainty"""
    #     q = k / N 
    #     return np.sqrt(q * (1 - q) / N)
        
    # def _get_poisson_err(self, k, N=None):
    #     """Calculate Poisson uncertainty
    #         Returns:
    #             absolute error if N=None
    #             relative error is N!=None
    #     """
    #     if N is None:
    #         N = 1
    #     return np.sqrt(k) / N 

    def get_poisson_bounds(self, k, confidence=0.68):
        """Get Poisson confidence intervals for rates"""
        alpha = 1 - confidence
        if k == 0:
            lower = 0
            upper = stats.chi2.ppf(confidence, 2) / 2
        else:
            lower = stats.chi2.ppf(alpha/2, 2*k) / 2
            upper = stats.chi2.ppf(1-alpha/2, 2*(k+1)) / 2
        
        return lower, upper # Return interval
        
    # def _get_wilson_err(self, k, N, z=1.0):
    #     """Calculate uncertainty with Wilson method"""
    #     alpha = 2 * (1 - norm.cdf(z)) # 68% confidence with z=1.0
    #     lower, upper = proportion_confint(k, N, alpha=alpha, method="wilson")
    #     return (upper - lower) / 2 # half the interval 

    def _get_wilson_bounds(self, k, N, z=1.0):
        """Calculate uncertainty with Wilson method"""
        if N == 0:
            return 0, 0
        alpha = 2 * (1 - norm.cdf(z)) # 68% confidence with z=1.0
        lower, upper = proportion_confint(k, N, alpha=alpha, method="wilson")
        return lower, upper # Return interval
        
    ####################################################
    # DataFrame methods
    ####################################################
    
    def _append_result(self, results, label, k, N, eff, eff_err_lo,
                       eff_err_hi, rate, rate_err_lo, rate_err_hi):
        """Add efficiency result to result list."""
        result = {
            "Type": label,
            "Events Passing (k)": k,
            "Total Events (N)": N,
            "Efficiency [%]": round(float(eff * 100), 3),
            "Efficiency Error Low [%]": round(float(eff_err_lo * 100), 3),
            "Efficiency Error High [%]": round(float(eff_err_hi * 100), 3),
            r"Rate [$\text{day}^{-1}$]": round(float(rate), 3) if rate is not None else None,
            r"Rate Error Low [$\text{day}^{-1}$]": round(float(rate_err_lo), 3) if rate_err_lo is not None else None,
            r"Rate Error High [$\text{day}^{-1}$]": round(float(rate_err_hi), 3) if rate_err_hi is not None else None
        }
        results.append(result)

    ####################################################
    # Histogram methods
    ####################################################
    
    def _get_N_from_hists(self, hists, selection, label): 
        """
        Retrieve number of events passing in a histogram
        """
        h = hists[selection] 
        h = h[{"selection": label}]  
        return int(h.sum())
        
    ####################################################
    # Efficiency from histograms methods 
    ####################################################

    def _get_rates(self, k, walltime_days):
        """Calculate rates and rate errors for given counts and walltime"""
        rate = k / walltime_days
        k_lo, k_hi = self.get_poisson_bounds(k)
        rate_err_lo = k_lo / walltime_days  
        rate_err_hi = k_hi / walltime_days
        return rate, rate_err_lo, rate_err_hi
    
    def _get_signal_eff_and_rate(self, hists, selection, title, generated_events, walltime_days, results):
        """Calculate signal efficiency for given selection"""
        k = self._get_N_from_hists(hists, selection, label="CE-like")   
        # Get efficiency
        eff = k / generated_events if int(float(generated_events)) > 0 else 0
        eff_err_lo, eff_err_hi = self._get_wilson_bounds(k, generated_events)
        # Get rates
        rate, rate_err_lo, rate_err_hi = self._get_rates(k, walltime_days)
        self._append_result(results, title, int(k), int(float(generated_events)), 
                           eff, eff_err_lo, eff_err_hi, rate, rate_err_lo, rate_err_hi)
    
    def _get_veto_eff_and_rate(self, hists, selection, title, walltime_days, results):
        """Calculate veto efficiency and rate for given selection"""
        k = self._get_N_from_hists(hists, selection, label="Unvetoed")    
        N = self._get_N_from_hists(hists, selection, label="CE-like")   
        # Get efficiency
        eff = (1 - k / N) if N > 0 else 0
        eff_err_lo, eff_err_hi = self._get_wilson_bounds(k, N)
        # Get rates
        rate, rate_err_lo, rate_err_hi = self._get_rates(k, walltime_days)
        # Store result
        self._append_result(results, title, int(k), int(N), eff, eff_err_lo, eff_err_hi, 
                           rate, rate_err_lo, rate_err_hi)
    
    def analyse_hists(
        self, 
        hists,
        livetime,
        on_spill,
        on_spill_frac, 
        generated_events, 
        veto
    ):
        """
        Report efficiency results for signal and veto analysis.
        
        Args:
            hists: Selection of histograms 
            generated_events (int, opt): Number of generated events in dataset. Defaults to 4e6.
            livetime (float): The total livetime in seconds for this dataset.
            on_spill (bool): Whether we are using on spill cuts. 
            on_spill_frac (float): The fraction of livetime in onspill. Defaults to 32.2%. 
            veto (bool): Whether to include veto analysis. Defaults to True. 
            
        Returns:
            pd.DataFrame: Results containing efficiency calculations
        """
        self.logger.log(f"Getting efficiency from histograms", "info")
    
        try:
            generated_events = float(generated_events)
        except (ValueError, TypeError):
            generated_events = 0
        
        # Get walltime
        if on_spill:
            walltime = livetime / on_spill_frac
        else:
            walltime = livetime / (1-on_spill_frac)
        
        # Convert to days
        walltime_days = walltime / (24*3600)
        
        # Init results
        results = []
        
        # Signal efficiency for CE selection over wide range
        self._get_signal_eff_and_rate(hists, "mom_full", "Signal (wide)", generated_events, walltime_days, results)
        self._get_signal_eff_and_rate(hists, "mom_ext", "Signal (ext)", generated_events, walltime_days, results)
        self._get_signal_eff_and_rate(hists, "mom_sig", "Signal (sig)", generated_events, walltime_days, results)
        
        # Veto efficiency and rates
        self._get_veto_eff_and_rate(hists, "mom_full", "Veto (wide)", walltime_days, results)
        self._get_veto_eff_and_rate(hists, "mom_ext", "Veto (ext)", walltime_days, results)
        self._get_veto_eff_and_rate(hists, "mom_sig", "Veto (sig)", walltime_days, results)
        
        self.logger.log("Returning efficiency information", "success")
        
        return pd.DataFrame(results)
        
    # With nested functions
    # def analyse_hists(
    #     self, 
    #     hists,
    #     livetime,
    #     on_spill,
    #     on_spill_frac, 
    #     generated_events, 
    #     veto
    # ):
    #     """
    #     Report efficiency results for signal and veto analysis.
        
    #     Args:
    #         hists: Selection of histograms 
    #         generated_events (int, opt): Number of generated events in dataset. Defaults to 4e6.
    #         livetime (float): The total livetime in seconds for this dataset.
    #         on_spill (bool): Whether we are using on spill cuts. 
    #         on_spill_frac (float): The fraction of livetime in onspill. Defaults to 32.2%. 
    #         veto (bool): Whether to include veto analysis. Defaults to True. 
            
    #     Returns:
    #         pd.DataFrame: Results containing efficiency calculations
    #     """
    #     self.logger.log(f"Getting efficiency from histogams", "info")
    
    #     try:
    #         generated_events = float(generated_events)
    #     except (ValueError, TypeError):
    #         generated_events = 0

    #     # Get walltime
    #     if on_spill:
    #         walltime = livetime / on_spill_frac
    #     else:
    #         walltime = livetime / (1-on_spill_frac)
    #     # Convert to days
    #     walltime_days = walltime / (24*3600)

    #     # Init results
    #     results = []

    #     def get_rates(k):
    #         # Get rates
    #         rate = k / walltime_days
    #         k_lo, k_hi = self.get_poisson_bounds(k)
    #         rate_err_lo = k_lo / walltime_days  
    #         rate_err_hi = k_hi / walltime_days
    #         return rate, rate_err_lo, rate_err_hi
            
    #     def get_signal_eff(selection, title): 
    #         k = self._get_N_from_hists(hists, selection, label="CE-like")   
    #          # Get efficiency
    #         eff = k / generated_events if int(float(generated_events)) > 0 else 0
    #         eff_err_lo, eff_err_hi = self._get_wilson_bounds(k, generated_events)
    #         # Get rates
    #         rate, rate_err_lo, rate_err_hi = get_rates(k)
    #         self._append_result(results, title, int(k), int(float(generated_events)), eff, eff_err_lo, eff_err_hi, rate, rate_err_lo, rate_err_hi)
    
    #     # Signal efficiency for CE selection over wide range
    #     get_signal_eff(selection="mom_full", title="Signal (wide)")
    #     get_signal_eff(selection="mom_ext", title="Signal (ext)")
    #     get_signal_eff(selection="mom_sig", title="Signal (sig)")

    #     def get_veto_eff_and_rate(selection, title): 
    #         k = self._get_N_from_hists(hists, selection, label="Unvetoed")    
    #         N = self._get_N_from_hists(hists, selection, label="CE-like")   
    #         # Get efficiency
    #         eff = (1 - k / N) if N > 0 else 0
    #         eff_err_lo, eff_err_hi = self._get_wilson_bounds(k, N)
    #         # Get rates
    #         rate, rate_err_lo, rate_err_hi = get_rates(k)
    #         # Store result
    #         self._append_result(results, title, int(k), int(N), eff, eff_err_lo, eff_err_hi, rate, rate_err_lo, rate_err_hi)
    
    #     # Signal efficiency for CE selection over wide range
    #     get_veto_eff_and_rate(selection="mom_full", title="Veto (wide)")
    #     get_veto_eff_and_rate(selection="mom_ext", title="Veto (ext)")
    #     get_veto_eff_and_rate(selection="mom_sig", title="Veto (sig)")

    #     self.logger.log("Returning efficiency information", "success")
        
    #     return pd.DataFrame(results)

    # ####################################################
    # # Efficiency from cuts DataFrame 
    # # HISTORICAL METHOD
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

