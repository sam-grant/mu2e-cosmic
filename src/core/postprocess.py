import awkward as ak
import sys
import yaml
import os

from pyutils.pylogger import Logger
from pyutils.pyselect import Select

sys.path.append("../utils")
from cut_manager import CutManager
from hist_analyser import HistAnalyser
from io_manager import load_config_yaml

class PostProcess():
    """Class for postprocessing
    """
    def __init__(self, on_spill, write_events=False, write_event_info=False, 
                 generated_events=None, livetime=None, 
                 on_spill_frac=None,
                 veto=True, verbosity=1):
        """Initialise

            Args:
                on_spill (bool): Whether we are using on spill cuts. Propagated from Process by run.py
                write_events (bool, opt): write filtered events. Defaults to False.
                write_event_info (bool, opt): write filtered event info. Defaults to False.
                generated_events (int, opt): Number of generated events. Defaults to value from config.
                livetime (float, opt): The total livetime in seconds for this dataset. Defaults to value from config.
                on_spill_frac (dict, opt): Fraction of walltime in single and two batch onspill. Defaults to values from config.
                veto (bool, opt): Whether running with CRV veto. Defaults to True.
                verbosity (int, opt): Printout level. Defaults to 1.
        """
        # Load defaults from config
        config = load_config_yaml("config/common/analysis.yaml", __file__)
        
        # Use config defaults if not provided
        if generated_events is None:
            generated_events = config["defaults"]["generated_events"]
        if livetime is None:
            livetime = config["defaults"]["livetime_seconds"]
        if on_spill_frac is None:
            on_spill_frac = config["timing"]["onspill_fractions"]
        # Member variables
        self.write_events = write_events
        self.write_event_info = write_event_info
        self.generated_events = generated_events
        self.livetime = livetime
        self.on_spill = on_spill
        self.on_spill_frac = on_spill_frac
        self.veto = veto
        self.verbosity = verbosity
        
        # Selector 
        self.selector = Select(verbosity=0)

        # Efficency and rates handler
        self.hist_analyser = HistAnalyser(verbosity=self.verbosity)

        # Start logger
        self.logger = Logger(
            print_prefix="[PostProcess]",
            verbosity=self.verbosity
        )

        self.logger.log(f"Initialised", "info")

    def combine_cut_flows(self, results, format_as_df=True):
        """
        Combine cuts flows from results list
        """

        # Check if we have results
        if not results:
            self.logger.log(f"results is None", "warning") 
            return None
            
        cut_flow_list = [] 
        if isinstance(results, list): 
            for result in results: 
                if "cut_flow" in result: 
                    cut_flow_list.append(result["cut_flow"])
        else: 
            cut_flow_list.append(results["cut_flow"])

        # Get cut manager
        cut_manager = CutManager()

        # Combine and format
        combined_cut_flow = cut_manager.combine_cut_flows(
            cut_flow_list = cut_flow_list, 
            format_as_df = format_as_df
        )

        # Round 
        combined_cut_flow = combined_cut_flow.round(3)

        return combined_cut_flow

    def combine_hists(self, results):
        """Combine histograms
        
        Args:
            results: List of results per file
            
        Returns:
            dict: Combined histograms
        """
        combined_hists = {}
        
        # Check if we have results
        if not results:
            self.logger.log(f"results is None", "warning") 
            return None
        
        # Loop through all files
        for result in results:
            # Skip if no histograms in this file
            if "hists" not in result or not result["hists"]:
                continue
            
            # Process each histogram type
            for hist_name, hist_obj in result["hists"].items():
                if hist_name not in combined_hists:
                    # First time seeing this histogram type, initialise
                    combined_hists[hist_name] = hist_obj.copy()
                else:
                    # Add this histogram to the accumulated one
                    combined_hists[hist_name] += hist_obj
                    
        self.logger.log(f"Combined {len(combined_hists)} histograms over {len(results)} results", "success")
        return combined_hists
        
    def combine_arrays(self, results):
        """Combine filtered arrays from multiple files
        Args: 
            results (list): List of results 
        """
        try:
            arrays = []
            
            # Check if we have results
            if not results:
                self.logger.log(f"results is None", "warning") 
                return None
                
            # Loop through all files
            for result in results: #
                array = ak.Array(result["events"])
                if len(array) == 0:
                    continue
                # Concatenate arrays
                arrays.append(array)
                
            if len(arrays) == 0:
                self.logger.log(f"Combined array has zero length", "warning") 
                return arrays
    
            combined_array = ak.concatenate(arrays)
            
            self.logger.log(f"Combined arrays, result contains {len(combined_array)} events", "success")
    
            return combined_array
        except Exception as e:
            self.logger.log(f"Exception while combining arrays: {e}", "error")
            return None

    def get_combined_analysis(self, hists): # , transpose=True):
        """
        Calculate combined analysis of efficiency and rates from histograms

        Args:
            hists: Combined histograms.
            transpose (bool, opt): Transpose the DataFrame. Defaults to True.
        Returns:
            pd.DataFrame of efficiency information
        """
        try:
            result = self.hist_analyser.analyse_hists(
                hists, 
                livetime=self.livetime,
                on_spill=self.on_spill,
                on_spill_frac=self.on_spill_frac,
                generated_events=self.generated_events,
                veto=self.veto
            )            
            # if transpose:
            #      # Transpose and reset index
            #     result = result.T.reset_index()
            #     # Replace headers with the first row
            #     result.columns = result.iloc[0]
            #     # Drop first redundant and reset indices again
            #     result = result.drop(0).reset_index(drop=True)

            # Just transpose - the index (row labels) automatically becomes a column
            # result = result.T
            # # Reset index to make the momentum windows (Wide, Extended, Signal) regular rows
            # result = result.reset_index()
            # # Rename the first column to something meaningful
            # result = result.rename(columns={'index': 'Window'})
            
            self.logger.log(f"Analysed histograms:", "success")
            return result
        except Exception as e:
            self.logger.log(f"Exception while analysing histograms: {e}", "error")
            return None

    # def get_combined_analysis(self, hists, transpose=True):
    #     """Calculate combined analysis of efficiency and rates from histograms"""
    #     try:
    #         result = self.hist_analyser.analyse_hists(
    #             hists, 
    #             livetime=self.livetime,
    #             on_spill=self.on_spill,
    #             on_spill_frac=self.on_spill_frac,
    #             generated_events=self.generated_events,
    #             veto=self.veto
    #         )
            
    #         print("Original DataFrame shape:", result.shape)
    #         print("Original DataFrame columns:", result.columns.tolist())
    #         print("Original DataFrame index:", result.index.tolist())
            
    #         if transpose:
    #             print("Before transpose:")
    #             print(result.head(3))
                
    #             # Just transpose
    #             result = result.T
    #             print("After transpose shape:", result.shape)
    #             print("After transpose columns:", result.columns.tolist())
    #             print("After transpose index:", result.index.tolist())
    #             print("After transpose:")
    #             print(result.head(3))
                
    #             # Reset index
    #             result = result.reset_index()
    #             print("After reset_index shape:", result.shape)
    #             print("After reset_index columns:", result.columns.tolist())
    #             print("Final result:")
    #             print(result.head())
                
    #             # Rename the first column
    #             result = result.rename(columns={'index': 'Window'})
                    
    #         self.logger.log(f"Analysed histograms:", "success")
    #         return result
    #     except Exception as e:
    #         self.logger.log(f"Exception while analysing histograms: {e}", "error")
    #         import traceback
    #         traceback.print_exc()
    #         return None
        
    def get_event_info(self, results): 
        """
        Get filtered event info

        Args: 
            results (list): list of results 
            out_path: File path for txt output 
        """

        # Check if we have results
        if not results:
            self.logger.log(f"results is None", "warning") 
            return None

        try:
            output = []
            count = 0
            
            for i, result in enumerate(results): 
                
                data = ak.Array(result["events"])
                
                if len(data) == 0:
                    continue
    
                # Get tracker middle times
                trk_mid = self.selector.select_surface(data["trkfit"], surface_name="TT_Mid")
                track_time = data["trkfit"]["trksegs"]["time"][trk_mid]
                # Get coinc entrance times
                coinc_time = data["crv"]["crvcoincs.time"]
                
                # Extract values
                track_time_str = "" 
                coinc_time_str = ""
                
                # Extract floats from track_time (nested structure: [[[values]], [[values]]])
                track_floats = []
                for nested in track_time:
                    for sublist in nested:
                        for val in sublist:
                            track_floats.append(float(val))
                
                # Extract floats from coinc_time (structure: [[], []])
                coinc_floats = []
                for sublist in coinc_time:
                    for val in sublist:
                        coinc_floats.append(float(val))
                
                # Format as strings with precision
                if track_floats:
                    track_time_str = ", ".join([f"{val:.6f}" for val in track_floats])
                
                if coinc_floats:
                    coinc_time_str = ", ".join([f"{val:.6f}" for val in coinc_floats])
            
                # Calculate dt
                dt_str = ""
                if track_floats and coinc_floats:
                    # Calculate dt between first track time and first coinc time
                    dt_value = abs(track_floats[0] - coinc_floats[0])
                    dt_str = f"{dt_value:.6f}"
                
                output.append(f"  Index:            {i}")
                output.append(f"  Subrun:           {data['evt']['subrun']}")
                output.append(f"  Event:            {data['evt']['event']}")
                output.append(f"  File:             {result['id']}")
                output.append(f"  Track time [ns]:  {track_time_str}") 
                output.append(f"  Coinc time [ns]:  {coinc_time_str if len(coinc_time_str)>0 else None}") 
                output.append(f"  dt [ns]:          {dt_str if len(str(dt_str))>0 else 'N/A'}")
                output.append("-" * 40)
    
                count += 1
            
            output = "\n".join(output)
    
            self.logger.log(f"Retrieved background event info", "success")
    
            return output 
        except Exception as e:
            self.logger.log(f"Exception while retrieving background event info: {e}", "error")
            return None

    def execute(self, results): 
        """ 
        Args: 
            results (list): list of results
        Returns:
            tuple of combined arrays and combined histograms
        """
        try:
            
            # This handles single files 
            if not isinstance(results, list):
                results = [results]
    
            cut_flow = self.combine_cut_flows(results)
            hists = self.combine_hists(results)
            analysis = self.get_combined_analysis(hists)
            events = self.combine_arrays(results) if self.write_events else None 
            event_info = self.get_event_info(results) if self.write_event_info else None 
    
            output = {
                "cut_flow": cut_flow,
                "hists": hists,
                "analysis": analysis,
                "events": events,
                "event_info": event_info
            }
            
            self.logger.log(f"Postprocessing complete:\n\tReturning dict of combined cut flows, histograms, filtered events, and filtered event info", "success")
            
            return output

        except Exception as e:
            self.logger.log(f"Error during postprocessing {e}", "error")
            raise e
