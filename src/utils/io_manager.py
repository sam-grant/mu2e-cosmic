import os
import h5py
import numpy as np
import pickle
import awkward as ak
import pyarrow.parquet as pq
import hist
import pandas as pd

from pyutils.pylogger import Logger

class Write:
    def __init__(self, out_path = "test_out", verbosity=1):
        """Write analysis outputs 
            Args: 
                out_path (str, opt): Output directory. Defaults to "test_out". 
                verbosity (int, opt): Printout level.         
        """
        self.out_path = out_path 
        # Create output directory if it doesn't exist
        os.makedirs(self.out_path, exist_ok=True)
        
        self.logger = Logger(
            print_prefix = "[Write]",
            verbosity=verbosity
        )

        self.logger.log(f"Initialised with out_path={out_path}", "success")

    def write_pkl(self, results, out_name="results.pkl"):
        """Save all analysis outputs to pickle.

        Note that pickle provides limited portability. 

        Args: 
            results (dict): Postprocessing output.
            out_name (str, opt): Output file name. Defaults to "results.pkl".
        """
        try: 
            out_path = os.path.join(self.out_path, out_name) 
            # Open file
            with open(out_path, "wb") as f:
                pickle.dump(results, f)
            # Confirm 
            self.logger.log(f"Wrote results {out_path}", "success")
        except Exception as e: 
            self.logger.log(f"Failed to write results to pickle: {e}", "error")      

    def _write_df_to_csv(self, df, out_name):
        """Helper to write DataFrame to CSV

        Args: 
            df (pd.DataFrame): DataFrame.
            out_name (str): Output file name. 
        """
        # Define output file 
        out_path = os.path.join(self.out_path, out_name)
        # Convert to DataFrame if not already one
        if not isinstance(df, pd.DataFrame):
           df = pd.DataFrame(df)
        # Open and write file
        df.to_csv(out_path, index=False)
        # Pass out_path back for logging
        return out_path
            
    def write_cut_flow_csv(self, df_cut_flow, out_name="cut_flow.csv"):
        """Save cut flow  to CSV

        Args: 
            df_cut_flow (pd.DataFrame): Cut flow.
            out_name (str, opt): Output file name. Defaults to "stats.csv". 
        """
        try:
            # Write
            out_path = self._write_df_to_csv(df_cut_flow, out_name)
            # Confirm
            self.logger.log(f"Wrote cut flow to {out_path}", "success")
        except Exception as e:
            self.logger.log(f"Failed to write cut flow: {e}", "error")
            raise

    def write_efficiency_csv(self, df_eff, out_name="efficiency.csv"):
        """Save efficiency info to CSV

        Args: 
            df_eff (pd.DataFrame): Efficiency info.
            out_name (str, opt): Output file name. Defaults to "efficiency.csv". 
        """
        try:
            # Write
            out_path = self._write_df_to_csv(df_eff, out_name)
            # Confirm
            self.logger.log(f"Wrote efficiency to {out_path}", "success")
        except Exception as e:
            self.logger.log(f"Failed to write efficiency info: {e}", "error")
            raise

    def write_hists_h5(self, hists, out_name="hists.h5"):
        """Save histogram selection to ROOT
            Args: 
                hists: Selection of histogram objects.
        """
        try:
            # Define output file 
            out_path = os.path.join(self.out_path, out_name)
            # Open file
            with h5py.File(out_path, "w") as f:
                # Iterate through dict
                for name, hist_obj in hists.items(): 
                    # Create group based on selection name
                    group = f.create_group(name)
                    # Save the values array
                    group.create_dataset("values", data=hist_obj.values())
                    # Save each axis
                    axes_group = group.create_group("axes")
                    
                    for i, axis in enumerate(hist_obj.axes):
                        axis_group = axes_group.create_group(f"axis_{i}")
                        # Save axis metadata
                        axis_group.attrs["name"] = getattr(axis, "name", f"axis_{i}")
                        axis_group.attrs["label"] = getattr(axis, "label", "")
                        axis_group.attrs["type"] = type(axis).__name__
                        
                        # Save axis-specific data
                        if type(axis).__name__ == "StrCategory": # this one uses strings to specify axes
                            categories = list(axis)
                            dt = h5py.string_dtype(encoding="utf-8")
                            axis_group.create_dataset("categories", data=categories, dtype=dt)
                        else: # For Regular, Variable, Boolean, and Integer, just save edges
                            axis_group.create_dataset("edges", data=axis.edges)\
                # Confirm
                self.logger.log(f"Wrote histograms to {out_path}", "success")
        except Exception as e: 
            self.logger.log(f"Failed to write histograms: {e}", "error")
            raise

    def write_events_parquet(self, events, out_name="events.parquet"):
        """Save filtered awkward array to Parquet
            Args: 
                data (ak.Array): Awkward array.
                out_name (str, opt): Output file name. Defaults to "events.parquet".
        
        """
        try:
            # Check if we have events
            if events is None or len(events) == 0:
                self.logger.log("No events to save", "warning")
                return
            # Define full output file path
            out_path = os.path.join(self.out_path, out_name)
            ak.to_parquet(events, out_path)
            self.logger.log(f"Wrote events {out_path}", "success")
        except Exception as e1:
            print(f"Error saving awkward array: {e1}")
            # Fallback
            try:
                self.logger.log("Attempting to save with arrow conversion", "info")
                table = ak.to_arrow_table(events)
                pq.write_table(table, out_path)
            except Exception as e2:
                self.logger.log(f"Fallback failed {e2}", "error")
                raise

    def write_info_txt(self, info, out_name="info.txt"):
        """Save background event info to txt
        Args: 
            info (str): Background event info.
            out_name (str, opt): Output file name. Defaults to "info.txt".
        """
        try:
            # Check if info exists
            if info is None or info == "":
                self.logger.log("No info to save", "warning")
                return 
            # Define full output file path
            out_path = os.path.join(self.out_path, out_name)
            # Open file
            with open(out_path, "w") as f:
                # Write file
                f.write(info)
            # Confirm
            self.logger.log(f"Wrote info to {out_path}", "success")
        except Exception as e:
            self.logger.log(f"Failed to write info: {e}", "error")

    def write_all(self, results):
        self.logger.log(f"Saving all", "info")
        
        self.logger.log(f"Writing all results to pickle", "info")
        self.write_pkl(results)
        
        self.logger.log(f"Writing cut flow to csv", "info")
        self.write_cut_flow_csv(results["cut_flow"])
        
        self.logger.log(f"Writing hists to h5", "info")
        self.write_hists_h5(results["hists"])

        self.logger.log(f"Writing efficiency info to csv", "info")
        self.write_efficiency_csv(results["efficiency"])
        
        self.logger.log(f"Writing background events to parquet", "info")
        self.write_events_parquet(results["events"])

        self.logger.log(f"Writing background info to txt", "info")
        self.write_info_txt(results["event_info"])

class Load:
    def __init__(self, in_path="test_out", verbosity=1):
        """Load analysis outputs 
            Args: 
                in_path (str, opt): Analysis output directory. Defaults to "test_out". 
                verbosity (int, opt): Printout level.         
        """
        self.in_path = in_path 
        self.logger = Logger(
            print_prefix = "[Load]",
            verbosity=verbosity
        )

        self.logger.log(f"Initialised with out_path={in_path}", "success")

    def _get_path(self, in_name):
        """Helper to get the full file path""" 
        # Return input file path
        return os.path.join(self.in_path, in_name)

    def load_pkl(self, in_name="results.pkl"):
        """Load all analysis outputs from pickle file.
        
        Note that pickle provides limited portability.
        
        Args:
            in_name (str, opt): Input file name. Defaults to "results.pkl".
            
        Returns:
            dict: Loaded postprocessing output, or None if loading failed.
        """
        try:
            # Get path
            in_path = self._get_path(in_name)

            # Open and load file
            with open(in_path, "rb") as f:
                results = pickle.load(f)
                
            # Confirm successful load
            self.logger.log(f"Successfully loaded results from {in_path}", "success")
            return results
            
        except Exception as e:
            self.logger.log(f"Failed to load results from pickle: {e}", "error")
            raise

    def load_array_parquet(self, in_name="events.parquet"):
        """Load awkward array from Parquet

        Args:
            in_name (str, opt): Input file name
        """
        try:
            # Get path
            in_path = self._get_path(in_name)
            
            # Check if file exists
            if not os.path.exists(in_path):
                self.logger.log(f"File not found: {in_path}", "warning")
                return None # expected if no events
                
            # Load array
            array = ak.from_parquet(in_path)
            # Confirm successful load
            self.logger.log(f"Successfully loaded ak.Array from {in_path}", "success")
            return array
        except Exception as e:
            print(f"Error loading awkward array: {e}")
            # Fallback: try with arrow
            try:
                table = pq.read_table(in_path)
                array = ak.from_arrow(table)
                # Confirm successful load
                self.logger.log(f"Successfully loaded hists from {in_path}", "success")
                return events
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise
                
    def load_hists_h5(self, in_name="hists.h5"):
        """Load histogram dictionary from HDF5"""
        try: 
            # Hist container
            hists = {}
            
            # Define input file path
            in_path = self._get_path(in_name) 
            
            # Check if file exists
            if not os.path.exists(in_path):
                self.logger.log(f"File not found: {in_path}", "error")
                raise
                
            # Open file
            with h5py.File(in_path, "r") as f:
                for name in f.keys():
                    # Open group 
                    group = f[name]
                    # Reconstruct axes
                    axes = []
                    axes_group = group["axes"]
                    # Iterate through group and load axis info
                    for i in range(len(axes_group.keys())):
                        axis_group = axes_group[f"axis_{i}"]
                        axis_type = axis_group.attrs["type"]
                        axis_name = axis_group.attrs["name"]
                        axis_label = axis_group.attrs["label"]
                        
                        if axis_type == "Regular":
                            edges = axis_group["edges"][()]
                            bins = len(edges) - 1
                            start = edges[0]
                            stop = edges[-1]
                            axis = hist.axis.Regular(bins, start, stop, name=axis_name, label=axis_label)
                            
                        elif axis_type == "StrCategory":
                            categories = [s.decode("utf-8") for s in axis_group["categories"]]
                            axis = hist.axis.StrCategory(categories, name=axis_name, label=axis_label)
                            
                        elif axis_type == "Variable":
                            edges = axis_group["edges"][()]
                            axis = hist.axis.Variable(edges, name=axis_name, label=axis_label)
                            
                        elif axis_type == "Integer":
                            edges = axis_group["edges"][()]
                            start = int(edges[0])
                            stop = int(edges[-1]) + 1
                            axis = hist.axis.Integer(start, stop, name=axis_name, label=axis_label)
                            
                        elif axis_type == "Boolean":
                            axis = hist.axis.Boolean(name=axis_name, label=axis_label)
                            
                        else:
                            self.logger.log(f"Unknown axis type: {axis_type}", "error")
                            return None
    
                        # Append axis
                        axes.append(axis)
                    
                    # Create histogram and set values
                    new_hist = hist.Hist(*axes)              # 1. Create empty histogram with the axes
                    values = group["values"][()]             # 2. Load the saved values array from HDF5
                    new_hist.view(flow=False)[...] = values  # 3. Fill the histogram with the values
                    hists[name] = new_hist                   # 4. Store in the dictionary

            self.logger.log(f"Loaded histograms from {in_path}", "success")
            self.logger.log(f"Histogram info:\n\t{hists}", "max")
            
            # Return
            return hists

        except Exception as e:
            self.logger.log(f"Failed to load histograms: {e}", "error")
            raise

    def _load_csv(self, in_name):
        """Load CSV to DataFrame.
        
        Args:
            in_name (str, opt): Input file name.
            
        Returns:
            pd.DataFrame
        """
        # Define input file path
        in_path = self._get_path(in_name)
        
        # Check if file exists
        if not os.path.exists(in_path):
            self.logger.log(f"File not found: {in_path}", "error")
            raise
            
        # Return DataFrame and path
        return pd.read_csv(in_path), in_path
            
    def load_cut_flow_csv(self, in_name="cut_flow.csv"):
        """Load cut stats info from CSV file.
        
        Args:
            in_name (str, opt): Input file name. Defaults to "stats.csv".
            
        Returns:
            pd.DataFrame: Loaded cut flow.
        """
        try:
            # Load
            df_cut_flow, in_path = self._load_csv(in_name)
            # Confirm successful load
            self.logger.log(f"Loaded cut stats from {in_path}", "success")
            return in_path
            
        except Exception as e:
            self.logger.log(f"Failed to load cut flow: {e}", "error")
            raise

    def load_efficiency_csv(self, in_name="efficiency.csv"):
        """Load efficiency info from CSV file.
        
        Args:
            in_name (str, opt): Input file name. Defaults to "efficiency.csv".
            
        Returns:
            pd.DataFrame: Loaded efficiency info.
        """
        try:
            # Load
            df_eff, in_path = self._load_csv(in_name)
            # Confirm successful load
            self.logger.log(f"Loaded efficiency info from {in_path}", "success")
            return df_eff
            
        except Exception as e:
            self.logger.log(f"Failed to efficiency info: {e}", "error")
            raise
            
    def load_info_txt(self, in_name="info.txt"):
        """Load info from plain text file.
        
        Args:
            in_name (str, opt): Input file name. Defaults to "info.txt".
            
        Returns:
            str: Loaded info string, or None if loading failed.
        """
        try:
            # Define input file path
            in_path = self._get_path(in_name)
            
            # Check if file exists
            if not os.path.exists(in_path):
                self.logger.log(f"File not found: {in_path}", "warning")
                return None
                
            # Load text file
            with open(in_path, "r", encoding="utf-8") as f:
                info = f.read()
            
            # Confirm successful load
            self.logger.log(f"Loaded info from {in_path}", "success")
            return info
            
        except Exception as e:
            self.logger.log(f"Failed to load info: {e}", "error")
            raise # no file is expected if no events

    def load_all(self):
        """Load all from persistent storage
            
        Returns:
            results (dict)
        """
        return {
            "cut_flow": self.load_cut_flow_csv(),
            "hists": self.load_hists_h5(), 
            "efficiency": self.load_efficiency_csv(),
            "events": self.load_array_parquet(),
            "event_info": self.load_info_txt()
        }
        