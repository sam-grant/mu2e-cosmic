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
    # Rates calculations methods
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
        """Get total events in a momentum window with no cuts
        """
        try:
            h = hists[momentum_window]
            # Use the "all" election category for livetime scaling
            h_all = h[{"selection": "All"}]
            return int(h_all.sum())
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

    # def _get_rate_results(self, k, rates_dict): 
    #     """ Get rates for both modes """

    #     if rates_dict is None:
    #         return {}
        
    #     rate_1batch = rates_dict["1batch"]["rate"]
    #     rate_err_lo_1batch = rates_dict["1batch"]["rate_err_lo"] - rate_1batch
    #     rate_err_hi_1batch = rates_dict["1batch"]["rate_err_hi"] - rate_1batch
        
    #     rate_2batch = rates_dict["2batch"]["rate"] 
    #     rate_err_lo_2batch = rates_dict["2batch"]["rate_err_lo"] - rate_2batch
    #     rate_err_hi_2batch = rates_dict["2batch"]["rate_err_hi"] - rate_2batch

    #     # FIXME FIXME FIXME FIXME FIXME FIXME
    #     # Hardcode Run-1 livetime as 3.46e6 seconds (from Natalie)
    #     # FIXME/TODO: verify this and put it in YAML (or an arg in PostProcess)
    #     run1_livetime_seconds = 3.46e6
    #     run1_livetime_ratio = getattr(self, '_current_scaled_livetime', 0) / run1_livetime_seconds if hasattr(self, '_current_scaled_livetime') else 0

    #     # Calculate rates per Run-1 
    #     if run1_livetime_ratio > 0:
    #         rate_1b_per_run1 = k / run1_livetime_ratio
    #         rate_2b_per_run1 = k / run1_livetime_ratio
            
    #         # Calculate uncertainties for Run-1 rates using Poisson bounds
    #         k_lo, k_hi = self.get_poisson_bounds(k)
    #         rate_1b_err_lo_per_run1 = k_lo / run1_livetime_ratio - rate_1b_per_run1
    #         rate_1b_err_hi_per_run1 = k_hi / run1_livetime_ratio - rate_1b_per_run1
    #         rate_2b_err_lo_per_run1 = rate_1b_err_lo_per_run1
    #         rate_2b_err_hi_per_run1 = rate_1b_err_hi_per_run1
    #     else:
    #         rate_1b_per_run1 = 0
    #         rate_2b_per_run1 = 0
    #         rate_1b_err_lo_per_run1 = 0
    #         rate_1b_err_hi_per_run1 = 0
    #         rate_2b_err_lo_per_run1 = 0
    #         rate_2b_err_hi_per_run1 = 0

    #     return {
    #         r"Rate 1B [$\text{day}^{-1}$]": float(rate_1batch),
    #         r"Rate 1B Err$-$ [$\text{day}^{-1}$]": float(rate_err_lo_1batch),
    #         r"Rate 1B Err$+$ [$\text{day}^{-1}$]": float(rate_err_hi_1batch),
    #         r"Rate 2B [$\text{day}^{-1}$]": float(rate_2batch),
    #         r"Rate 2B Err$-$ [$\text{day}^{-1}$]": float(rate_err_lo_2batch),
    #         r"Rate 2B Err$+$ [$\text{day}^{-1}$]": float(rate_err_hi_2batch),
    #         r"Run-1 Livetimes": float(run1_livetime_ratio),
    #         r"Rate 1B [$\text{Run-1}^{-1}$]": float(rate_1b_per_run1),
    #         r"Rate 1B Err$-$ [$\text{Run-1}^{-1}$]": float(rate_1b_err_lo_per_run1),
    #         r"Rate 1B Err$+$ [$\text{Run-1}^{-1}$]": float(rate_1b_err_hi_per_run1),
    #         r"Rate 2B [$\text{Run-1}^{-1}$]": float(rate_2b_per_run1),
    #         r"Rate 2B Err$-$ [$\text{Run-1}^{-1}$]": float(rate_2b_err_lo_per_run1),
    #         r"Rate 2B Err$+$ [$\text{Run-1}^{-1}$]": float(rate_2b_err_hi_per_run1)
    #     }
        
    ####################################################
    # DataFrame methods
    ####################################################

    # def _append_result(self, results, label, k, N, eff, eff_err_lo, eff_err_hi, rates_dict=None):
    #     """Add efficiency and rate result with batch mode information"""
    #     # Get efficiency results
    #     eff_results = {
    #         "Type": label,
    #         "Passing": k,
    #         "Initial": N,
    #         "Eff [%]": float(eff * 100),
    #         r"Eff Err$-$ [%]": float((eff_err_lo - eff) * 100),
    #         r"Eff Err$+$ [%]": float((eff_err_hi - eff) * 100)
    #     }
    #     # Get rates results
    #     rates_results = self._get_rate_results(k, rates_dict) # Returns {} for signal datasets
    #     # Combine 
    #     results_dict = eff_results | rates_results
    #     # Append
    #     results.append(results_dict)

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
    
    # def _get_signal_eff_and_rate(self, hists, selection, title, generated_events, walltime_days_dict, results, veto):
    #     """Calculate signal efficiency for given selection"""
    #     k = self._get_N_from_hists(hists, selection, label="Select")   
        
    #     # Get efficiency
    #     eff = k / generated_events if int(float(generated_events)) > 0 else 0
    #     eff_err_lo, eff_err_hi = self._get_wilson_bounds(k, generated_events)
        
    #     # Get rates for all batch modes
    #     rates_dict = self.get_rates(k, walltime_days_dict) if veto else {}
        
    #     # Store result
    #     self._append_result(
    #         results, title, int(k), int(float(generated_events)), 
    #         eff, eff_err_lo, eff_err_hi, rates_dict
    #     )

    # def _get_signal_eff(self, hists, selection, title, generated_events, results):
    #     """Calculate signal efficiency for given selection
    #     Do not calculate rates for signal-only 
    #     """
    #     k = self._get_N_from_hists(hists, selection, label="Select")   
        
    #     # Get efficiency
    #     eff = k / generated_events if int(float(generated_events)) > 0 else 0
    #     eff_err_lo, eff_err_hi = self._get_wilson_bounds(k, generated_events)

    #     # Get rates for all batch modes
    #     rates_dict = None # (not used for signal datasets)
        
    #     # Store result
    #     self._append_result(
    #         results, title, int(k), int(float(generated_events)), 
    #         eff, eff_err_lo, eff_err_hi, rates_dict
    #     )

    
    # def _get_veto_eff_and_rate(self, hists, selection, title, walltime_days_dict, results):
    #     """Calculate veto efficiency and rate for given selection"""
    #     k = self._get_N_from_hists(hists, selection, label="Unvetoed")    
    #     N = self._get_N_from_hists(hists, selection, label="Select")   
        
    #     # Get efficiency
    #     eff = (1 - k / N) if N > 0 else 0
    #     k_bound_lo, k_bound_hi = self._get_wilson_bounds(k, N)
    #     # Transform 
    #     eff_err_lo = 1 - k_bound_hi  
    #     eff_err_hi = 1 - k_bound_lo  
        
    #     # Get rates for all batch modes
    #     rates_dict = self.get_rates(k, walltime_days_dict)
        
    #     # Store result
    #     self._append_result(
    #         results, title, int(k), int(N),
    #         eff, eff_err_lo, eff_err_hi, rates_dict
    #     )
    
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
        
        # Define momentum windows
        momentum_windows = [
            ("mom_full", "Wide"),
            ("mom_ext", "Extended"), 
            ("mom_sig", "Signal")
        ]
        
        # Collect data for each window
        results = {}
        
        for mom_window, col_name in momentum_windows:
            # Get event counts
            select_count = self._get_N_from_hists(hists, mom_window, label="Select")
            unvetoed_count = self._get_N_from_hists(hists, mom_window, label="Unvetoed") if veto else 0
            
            # Calculate efficiencies
            select_eff = select_count / generated_events if generated_events > 0 else 0
            select_eff_lo, select_eff_hi = self._get_wilson_bounds(select_count, generated_events)
            
            veto_eff = (1 - unvetoed_count / select_count) if select_count > 0 and veto else 0
            if veto and select_count > 0:
                unvet_lo, unvet_hi = self._get_wilson_bounds(unvetoed_count, select_count)
                veto_eff_lo = 1 - unvet_hi
                veto_eff_hi = 1 - unvet_lo
            else:
                veto_eff_lo = veto_eff_hi = 0
            
            # Calculate rates if doing veto analysis
            if veto and livetime_valid:
                scaled_livetime = self._scale_livetime_for_window(livetime, hists, mom_window, reference_window)
                self._current_scaled_livetime = scaled_livetime
                
                # Get walltime in days for both batch modes
                sec2day = 1 / (24*3600)
                walltime_days = {}
                for batch_mode, frac in on_spill_frac.items():
                    if on_spill: 
                        walltime = scaled_livetime / frac
                    else:
                        walltime = scaled_livetime / (1-frac)    
                    walltime_days[batch_mode] = walltime * sec2day
                
                # Calculate rates for both batch modes
                rates_dict = self.get_rates(unvetoed_count, walltime_days)
                
                # 1-batch rates
                rate_1b = rates_dict["1batch"]["rate"]
                rate_1b_lo = rates_dict["1batch"]["rate_err_lo"] - rate_1b
                rate_1b_hi = rates_dict["1batch"]["rate_err_hi"] - rate_1b
                
                # 2-batch rates
                rate_2b = rates_dict["2batch"]["rate"]
                rate_2b_lo = rates_dict["2batch"]["rate_err_lo"] - rate_2b
                rate_2b_hi = rates_dict["2batch"]["rate_err_hi"] - rate_2b
                
                # Run-1 rates (same for both batch modes)
                run1_livetime_seconds = 3.46e6
                run1_livetime_ratio = scaled_livetime / run1_livetime_seconds
                if run1_livetime_ratio > 0:
                    rate_run1 = unvetoed_count / run1_livetime_ratio
                    k_lo, k_hi = self.get_poisson_bounds(unvetoed_count)
                    rate_run1_lo = k_lo / run1_livetime_ratio - rate_run1
                    rate_run1_hi = k_hi / run1_livetime_ratio - rate_run1
                else:
                    rate_run1 = rate_run1_lo = rate_run1_hi = 0
                    
            else:
                rate_1b = rate_1b_lo = rate_1b_hi = 0
                rate_2b = rate_2b_lo = rate_2b_hi = 0
                rate_run1 = rate_run1_lo = rate_run1_hi = 0
                run1_livetime_ratio = 0
                walltime_days = {"1batch": 0, "2batch": 0}
            
            # Store data for this column
            results[col_name] = {
                "Generated": int(generated_events),
                "Selected": int(select_count),
                "Unvetoed": int(unvetoed_count) if veto else None,
                "Selection Eff [%]": float(select_eff * 100),
                r"Selection Eff Err$-$ [%]": float((select_eff_lo - select_eff) * 100),
                r"Selection Eff Err$+$ [%]": float((select_eff_hi - select_eff) * 100),
                "Veto Eff [%]": float(veto_eff * 100) if veto else None,
                "Veto Eff Err$-$ [%]": float((veto_eff_lo - veto_eff) * 100) if veto else None,
                "Veto Eff Err$+$ [%]": float((veto_eff_hi - veto_eff) * 100) if veto else None,
                "Livetime [days]": float(scaled_livetime / 3600) if veto else None,
                r"Rate 1B [$\text{day}^{-1}$]": float(rate_1b) if veto else None,
                r"Rate 1B Err- [$\text{day}^{-1}$]": float(rate_1b_lo) if veto else None,
                r"Rate 1B Err+ [$\text{day}^{-1}$]": float(rate_1b_hi) if veto else None,
                r"Rate 2B [$\text{day}^{-1}$]": float(rate_2b) if veto else None,
                r"Rate 2B Err$-$ [$\text{day}^{-1}$]": float(rate_2b_lo) if veto else None,
                r"Rate 2B Err$+$ [$\text{day}^{-1}$]": float(rate_2b_hi) if veto else None,
                "Run-1 Livetimes": run1_livetime_ratio if veto else None,
                r"Rate 1B [$\text{Run-1}^{-1}$]": float(rate_run1) if veto else None,
                r"Rate 1B Err$-$ [$\text{Run-1}^{-1}$]": float(rate_run1_lo) if veto else None,
                r"Rate 1B Err$+$ [$\text{Run-1}^{-1}$]": float(rate_run1_hi) if veto else None,
                r"Rate 2B [$\text{Run-1}^{-1}$]": float(rate_run1) if veto else None,  # Same as 1B for Run-1
                r"Rate 2B Err$-$ [$\text{Run-1}^{-1}$]": float(rate_run1_lo) if veto else None,
                r"Rate 2B Err$+$ [$\text{Run-1}^{-1}$]": float(rate_run1_hi) if veto else None,
            }
        
        # Create DataFrame from the results
        df = pd.DataFrame(results, )

        # Remove rows with all None values (for non-veto analysis)
        if not veto:
            df = df.dropna(how="all")

        # # FIXME
        # # Attempting correct formatting
        # # So awful
        # df.style.format({
        #     "Generated": "{:d}",
        #     "Selected": "{:d}",
        #     "Unvetoed": "{:d}",
        
        #     "Selection Eff [%]": "{:.3f}",
        #     "Selection Eff Err$-$ [%]": "{:.3f}",
        #     "Selection Eff Err$+$ [%]": "{:.3f}",
        
        #     "Veto Eff [%]": "{:.3f}",
        #     r"Veto Eff Err$-$ [%]": "{:.3f}",
        #     r"Veto Eff Err$+$ [%]": "{:.3f}",
        
        #     "Livetime [days]": "{:.2f}",
        
        #     "Rate 1B [$\\text{day}^{-1}$]": "{:.2f}",
        #     "Rate 1B Err$-$ [$\\text{day}^{-1}$]": "{:.2f}",
        #     "Rate 1B Err$+$ [$\\text{day}^{-1}$]": "{:.2f}",
        
        #     "Rate 2B [$\\text{day}^{-1}$]": "{:.2f}",
        #     "Rate 2B Err$-$ [$\\text{day}^{-1}$]": "{:.2f}",
        #     "Rate 2B Err$+$ [$\\text{day}^{-1}$]": "{:.2f}",
        
        #     "Run-1 Livetimes": "{:.3f}",
        
        #     r"Rate 1B [$\text{Run-1}^{-1}$]": "{:.2f}",
        #     r"Rate 1B Err$-$ [$\text{Run-1}^{-1}$]": "{:.2f}",
        #     r"Rate 1B Err$+$ [$\text{Run-1}^{-1}$]": "{:.2f}",
        
        #     r"Rate 2B [$\text{Run-1}^{-1}$]": "{:.2f}",
        #     r"Rate 2B Err$-$ [$\text{Run-1}^{-1}$]": "{:.2f}",
        #     r"Rate 2B Err$+$ [$\text{Run-1}^{-1}$]": "{:.2f}",
        # })

        
        self.logger.log("Completed analysis", "success")
        
        return df 