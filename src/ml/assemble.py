# Sam Grant 2025
# 

# System tools  
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore") # suppress warnings

# Data tools
import numpy as np
import pandas as pd
import awkward as ak


# Internal modules 
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(script_dir, "..", "utils"),
    os.path.join(script_dir, "..", "core")
])

from draw import Draw
from load import LoadML
from pyutils.pylogger import Logger

class AssembleDataset():
    """ Assemble and validate data from processing 
    Not really sure what to call this, 'preprocesing' would get confusing 
    """
    def __init__(self, run="i", cutset_name="dev", verbosity=1):
        logger = Logger(print_prefix="[Assemble]", verbosity=verbosity)
        self.run = run
        self.cutset_name = cutset_name
        self.base_in_path = Path(f"../../output/ml/{self.run}/data/")
        # A bit hacky but...
        # this is defined from the notebook/ml location
        self.img_out_path = Path(f"../../../output/images/ml/{self.run}/process/") 
        self.cry_data, self.ce_mix_data = LoadML(run=self.run).get_results()
        # Convert to DataFrame
        self.df_cry = ak.to_dataframe(self.cry_data["events"])
        self.df_ce_mix = ak.to_dataframe(self.ce_mix_data["events"])
        logger.log(f"Initialised", "success")

    def draw_hists(self):
        draw = Draw(cutset_name=self.cutset_name)
        draw.plot_summary(self.cry_data["hists"], out_path = self.img_out_path / "h1o_3x3_cuts_CRY.png")
        draw.plot_summary(self.ce_mix_data["hists"], out_path = self.img_out_path / "h1o_3x3_cuts_CE_mix.png")

    def get_cut_flows(self):
        return self.cry_data["cut_flow"], self.ce_mix_data["cut_flow"]

    def check_dT_window_results(self, dT_min = 0, dT_max = 150):
        
        def _apply_dT_window(df):

            dT = df["dT"].values            
            # Veto if dT is valid and in window
            veto_condition = (~np.isnan(dT)) & (dT >= dT_min) & (dT <= dT_max)

            # Event is vetoed if ANY coincidence passes cut
            df_with_veto = df.copy()
            df_with_veto["vetoed"] = veto_condition

            # For CE Mix2BB: deadtime (fraction of events incorrectly vetoed)
            # mix_in_window = (df_ce_mix['dT'] >= dt_min) & (df_ce_mix['dT'] <= dt_max)
            # mix_vetoed_events = df_ce_mix[mix_in_window].groupby(['event', 'subrun']).size()
            # mix_total_events = df_ce_mix.groupby(['event', 'subrun']).size()
            # eff_mix = len(mix_vetoed_events) / len(mix_total_events)
        
            event_veto = df_with_veto.groupby(["subrun", "event"])["vetoed"].max().reset_index()
            # total = df_with_veto.groupby(['event', 'subrun']).size()

            total = len(event_veto)
            unvetoed = total - event_veto["vetoed"].sum()

            return [total, unvetoed, f"{(100*(1-(unvetoed/total))):.2f}"]
    
        return pd.DataFrame({
            "Metric": ["Total", "Unvetoed", "Fraction [%]"],
            "CRY": _apply_dT_window(self.df_cry),
            "CE Mix": _apply_dT_window(self.df_ce_mix),
        })
    
    def get_ml_data(self):
        """ Load and prepare data for ML training/testing """
        # Get awk arrays
        cry_data, ce_mix_data = self.get_results()

        # SORT by event IDs to ensure reproducibility
        df_cry = df_cry.sort_values(["subrun", "event"]).reset_index(drop=True)
        df_ce_mix = df_ce_mix.sort_values(["subrun", "event"]).reset_index(drop=True)

        # Add labels
        df_cry["label"] = 1 # "signal"
        df_ce_mix["label"] = 0 # "background"

        self.logger.log("Got sorted and labelled DataFrames", "max")
        
        # Combine and shuffle
        df_train = pd.concat([df_cry, df_ce_mix], ignore_index=True)
        df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
        
        self.logger.log("Got combined training dataset", "max") 

        # Make a copy for later
        # We will want to be able to peek at full events post processing
        df_train_full = df_train.copy()

        self.logger.log(f"Columns: {df_train_full.columns}", "max")

        # Define columns to drop for training (but keep event/subrun for now)
        col_to_drop = ["d0", "tanDip", "maxr", "mom_mag", "PEs_per_hit", "t0", "timeStart", "timeEnd"]
        df_train.drop(columns=col_to_drop, inplace=True)

        # Separate features, labels, and metadata
        X = df_train.drop(["label", "event", "subrun"], axis=1)
        y = df_train["label"]
        metadata = df_train[["event", "subrun"]]

        # Split data AND metadata together
        X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
                X, y, metadata,
                test_size=0.2, 
                random_state=42,
                stratify=y
                )       
        
        self.logger.log("Split data", "max") 

        # Scaling 
        # Actually not really needed for BDTs
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.logger.log("Features scaled", "max") 

        self.logger.log("Got ML data", "success") 

        # Return as dictionary
        return {
                "df_train_full": df_train_full,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_train": y_train,
                "metadata_train": metadata_train,
                "metadata_test": metadata_test,
                }