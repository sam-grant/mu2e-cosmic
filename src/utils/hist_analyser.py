import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm
from scipy import stats
from pyutils.pylogger import Logger

class HistAnalyser():
    """
    Tools for histograms-based statisical analysis
    
    Efficiency and rate calculations with livetime scaling
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

    def _get_wilson_bounds(self, k, N, z=1.0):
        """Calculate uncertainty with Wilson method"""
        if N == 0:
            return 0, 1.0
        alpha = 2 * (1 - norm.cdf(z)) # 68% confidence with z=1.0
        lower, upper = proportion_confint(k, N, alpha=alpha, method="wilson")
        return lower, upper # Return interval
        
    ####################################################
    # Livetime scaling methods
    ####################################################
    
    def _get_livetime_scaling_factor(self, hists, momentum_window, reference_window="mom_sig"):
        """
        Calculate livetime scaling factor for different momentum windows
        
        Args:
            hists: Selection of histograms
            momentum_window (str): The momentum window to scale ("mom_full", "mom_ext", "mom_sig")
            reference_window (str): Reference window (signal region, default "mom_sig")
            
        Returns:
            float: Scaling factor (events_in_window / events_in_reference)
        """
        # Get total events in each window (sum over all selection types)
        events_reference = self._get_total_events_in_window(hists, reference_window)
        events_window = self._get_total_events_in_window(hists, momentum_window)
        
        if events_reference == 0:
            self.logger.log(f"Zero events in reference window {reference_window}", "warning")
            return 1.0
            
        scaling_factor = events_window / events_reference
        
        self.logger.log(f"Livetime scaling for {momentum_window}: {scaling_factor:.4f} "
                       f"({events_window}/{events_reference})", "info")
        
        return scaling_factor
    
    def _get_total_events_in_window(self, hists, momentum_window):
        """Get total events in a momentum window across all selection types"""
        try:
            h = hists[momentum_window]
            return int(h.sum())
        except KeyError:
            self.logger.log(f"Momentum window {momentum_window} not found in histograms", "error")
            return 0
    
    def _scale_livetime_for_window(self, livetime, hists, momentum_window, reference_window="mom_sig"):
        """
        Scale livetime for a specific momentum window
        
        Args:
            livetime (float): Reference livetime (for signal region)
            hists: Selection of histograms
            momentum_window (str): Window to scale livetime for
            reference_window (str): Reference window (signal region)
            
        Returns:
            float: Scaled livetime for the momentum window
        """
        if momentum_window == reference_window:
            return livetime
            
        scaling_factor = self._get_livetime_scaling_factor(hists, momentum_window, reference_window)
        scaled_livetime = livetime * scaling_factor
        
        return scaled_livetime

    ####################################################
    # DataFrame methods
    ####################################################

    def _append_result(self, results, label, k, N, eff, eff_err_lo, eff_err_hi, rates_dict):
        """Add efficiency and rate result with batch mode information"""
        
        # Get rates for both modes
        rate_1batch = rates_dict["1batch"]["rate"]
        rate_err_lo_1batch = rates_dict["1batch"]["rate_err_lo"] - rate_1batch
        rate_err_hi_1batch = rates_dict["1batch"]["rate_err_hi"] - rate_1batch
        
        rate_2batch = rates_dict["2batch"]["rate"] 
        rate_err_lo_2batch = rates_dict["2batch"]["rate_err_lo"] - rate_2batch
        rate_err_hi_2batch = rates_dict["2batch"]["rate_err_hi"] - rate_2batch
        
        # Hardcode Run-1 livetime as 3.46e6 seconds (from Natalie)
        # TODO: verify this and put it in YAML (or an arg in PostProcess)
        run1_livetime_seconds = 3.46e6
        run1_livetime_ratio = getattr(self, '_current_scaled_livetime', 0) / run1_livetime_seconds if hasattr(self, '_current_scaled_livetime') else 0

        # Calculate rates per Run-1 
        if run1_livetime_ratio > 0:
            rate_1b_per_run1 = k / run1_livetime_ratio
            rate_2b_per_run1 = k / run1_livetime_ratio
            
            # Calculate uncertainties for Run-1 rates using Poisson bounds
            k_lo, k_hi = self.get_poisson_bounds(k)
            rate_1b_err_lo_per_run1 = k_lo / run1_livetime_ratio - rate_1b_per_run1
            rate_1b_err_hi_per_run1 = k_hi / run1_livetime_ratio - rate_1b_per_run1
            rate_2b_err_lo_per_run1 = rate_1b_err_lo_per_run1
            rate_2b_err_hi_per_run1 = rate_1b_err_hi_per_run1
        else:
            rate_1b_per_run1 = 0
            rate_2b_per_run1 = 0
            rate_1b_err_lo_per_run1 = 0
            rate_1b_err_hi_per_run1 = 0
            rate_2b_err_lo_per_run1 = 0
            rate_2b_err_hi_per_run1 = 0
    
        result = {
            "Type": label,
            "k": k,
            "N": N,
            "Eff [%]": float(eff * 100),
            r"Eff Err$-$ [%]": float((eff_err_lo - eff) * 100),
            r"Eff Err$+$ [%]": float((eff_err_hi - eff) * 100),
            r"Rate 1B [$\text{day}^{-1}$]": float(rate_1batch),
            r"Rate 1B Err$-$ [$\text{day}^{-1}$]": float(rate_err_lo_1batch),
            r"Rate 1B Err$+$ [$\text{day}^{-1}$]": float(rate_err_hi_1batch),
            r"Rate 2B [$\text{day}^{-1}$]": float(rate_2batch),
            r"Rate 2B Err$-$ [$\text{day}^{-1}$]": float(rate_err_lo_2batch),
            r"Rate 2B Err$+$ [$\text{day}^{-1}$]": float(rate_err_hi_2batch),
            r"Run-1 Livetimes": float(run1_livetime_ratio),
            r"Rate 1B [$\text{Run-1}^{-1}$]": float(rate_1b_per_run1),
            r"Rate 1B Err$-$ [$\text{Run-1}^{-1}$]": float(rate_1b_err_lo_per_run1),
            r"Rate 1B Err$+$ [$\text{Run-1}^{-1}$]": float(rate_1b_err_hi_per_run1),
            r"Rate 2B [$\text{Run-1}^{-1}$]": float(rate_2b_per_run1),
            r"Rate 2B Err$-$ [$\text{Run-1}^{-1}$]": float(rate_2b_err_lo_per_run1),
            r"Rate 2B Err$+$ [$\text{Run-1}^{-1}$]": float(rate_2b_err_hi_per_run1),
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

    def get_rates(self, k, walltime_days_dict):
        """Calculate rates and rate errors for given counts and walltime"""
        rates = {}
        for batch_mode, walltime_days in walltime_days_dict.items():
            rate = k / walltime_days
            k_lo, k_hi = self.get_poisson_bounds(k)
            rate_err_lo = k_lo / walltime_days  
            rate_err_hi = k_hi / walltime_days
            
            rates[batch_mode] = {
                "rate": rate,
                "rate_err_lo": rate_err_lo,
                "rate_err_hi": rate_err_hi
            }
            
        return rates
    
    def _get_signal_eff_and_rate(self, hists, selection, title, generated_events, walltime_days_dict, results):
        """Calculate signal efficiency for given selection"""
        k = self._get_N_from_hists(hists, selection, label="CE-like")   
        
        # Get efficiency
        eff = k / generated_events if int(float(generated_events)) > 0 else 0
        eff_err_lo, eff_err_hi = self._get_wilson_bounds(k, generated_events)
        
        # Get rates for all batch modes
        rates_dict = self.get_rates(k, walltime_days_dict)
        
        # Store result
        self._append_result(
            results, title, int(k), int(float(generated_events)), 
            eff, eff_err_lo, eff_err_hi, rates_dict
        )
    
    def _get_veto_eff_and_rate(self, hists, selection, title, walltime_days_dict, results):
        """Calculate veto efficiency and rate for given selection"""
        k = self._get_N_from_hists(hists, selection, label="Unvetoed")    
        N = self._get_N_from_hists(hists, selection, label="CE-like")   
        
        # Get efficiency
        eff = (1 - k / N) if N > 0 else 0
        k_bound_lo, k_bound_hi = self._get_wilson_bounds(k, N)
        # Transform 
        eff_err_lo = 1 - k_bound_hi  
        eff_err_hi = 1 - k_bound_lo  
        
        # Get rates for all batch modes
        rates_dict = self.get_rates(k, walltime_days_dict)
        
        # Store result
        self._append_result(
            results, title, int(k), int(N),
            eff, eff_err_lo, eff_err_hi, rates_dict
        )
    
    def analyse_hists(
        self, 
        hists,
        livetime,
        on_spill,
        on_spill_frac,
        generated_events, 
        veto,
        reference_window="mom_sig"
    ):
        """
        Report efficiency results for signal and veto analysis with livetime scaling.
        
        Args:
            hists: Selection of histograms 
            generated_events (int, opt): Number of generated events in dataset. Defaults to 4e6.
            livetime (float): The total livetime in seconds for the REFERENCE window (signal region).
            on_spill (bool): Whether we are using on spill cuts. 
            on_spill_frac (dict): The fraction of livetime in onspill single and two batch modes. 
            veto (bool): Whether to include veto analysis. Defaults to True.
            reference_window (str): Reference momentum window for livetime scaling (default "mom_sig")
            
        Returns:
            pd.DataFrame: Results containing efficiency calculations
        """
        self.logger.log(f"Getting efficiency from histograms with livetime scaling", "info")
    
        try:
            generated_events = float(generated_events)
        except (ValueError, TypeError):
            generated_events = 0

        # Check if livetime is valid 
        livetime_valid = livetime is not None and livetime > 0
        
        if not livetime_valid:
            self.logger.log(f"Livetime is 0 or None with veto={veto}", "warning")

        # Init results
        results = []
        
        # Define momentum windows and their titles
        momentum_windows = [
            ("mom_full", "CE-like (wide)"),
            ("mom_ext", "CE-like (ext)"), 
            ("mom_sig", "CE-like (sig)")
        ]
        
        # Process each momentum window
        for mom_window, title in momentum_windows:
            
            if livetime_valid:
                # Scale livetime for this momentum window
                scaled_livetime = self._scale_livetime_for_window(livetime, hists, mom_window, reference_window)
                
                # Store scaled livetime for Run-1 calculation in _append_result
                self._current_scaled_livetime = scaled_livetime
                
                # Get walltime in days
                walltime_days = {}
                sec2day = 1 / (24*3600)
                for batch_mode, frac in on_spill_frac.items():
                    if on_spill: 
                        walltime = scaled_livetime / frac
                    else:
                        walltime = scaled_livetime / (1-frac)    
                    walltime_days[batch_mode] = walltime * sec2day
            else:
                # Set dummy walltime for cases where we don't need rates
                walltime_days = {"1batch": 1.0, "2batch": 1.0}  # Will result in rate = k
                self._current_scaled_livetime = 0
            
            # Signal efficiency for this momentum window
            self._get_signal_eff_and_rate(hists, mom_window, title, generated_events, walltime_days, results)
            
            # Veto efficiency and rates for this momentum window
            if veto:
                veto_title = f"No veto ({title.split('(')[1]}"  # Extract the momentum range part: full, ext, sig
                self._get_veto_eff_and_rate(hists, mom_window, veto_title, walltime_days, results)
        
        self.logger.log("Returning efficiency information with scaled livetimes", "success")
        
        return pd.DataFrame(results)