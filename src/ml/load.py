# System tools  
import os
import sys
from pathlib import Path

# Python stack 
import pandas as pd
import awkward as ak

# ML tools
from sklearn.model_selection import train_test_split

# Internal modules 
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(script_dir, "..", "utils"),
    os.path.join(script_dir, "..", "core")
])
from io_manager import Load

class LoadML():
    """ Load up ML dataset """
    def __init__(self, run="h"):
        self.run = run
        self.base_in_path = Path(f"../../output/ml/{self.run}/data/")
        
    def get_full_results(self):
        """ Raw full processing output from pkl files
        Can use be used seperately for validation
        without cluttering main ML notebook """ 
        # Define input paths
        cry_in_path = self.base_in_path / "CRY_onspill-LH_aw"
        ce_mix_in_path = self.base_in_path/ "CE_mix2BB_onspill-LH_aw"

        # Load data
        cry_data = Load(in_path=cry_in_path).load_pkl()
        ce_mix_data = Load(in_path=ce_mix_in_path).load_pkl()

        return cry_data, ce_mix_data

    def get_ml_data(self):
        """ Load and prepare data for ML training/testing """
        # Get awk arrays
        cry_data, ce_mix_data = self.get_full_results()
    
        # Convert to DataFrame
        df_cry = ak.to_dataframe(cry_data["events"])
        df_ce_mix = ak.to_dataframe(ce_mix_data["events"])

        # SORT by event IDs to ensure reproducibility
        df_cry = df_cry.sort_values(['subrun', 'event']).reset_index(drop=True)
        df_ce_mix = df_ce_mix.sort_values(['subrun', 'event']).reset_index(drop=True)
        
        # Add labels
        df_cry["label"] = 1 # "signal"
        df_ce_mix["label"] = 0 # "background"

        # Combine and shuffle
        df_train = pd.concat([df_cry, df_ce_mix], ignore_index=True)
        df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

        # Make a copy for later
        # We will want to be able to peek at full events post processing
        df_train_full = df_train.copy()

        # Define columns to drop for training (but keep event/subrun for now)
        col_to_drop = ["d0", "tanDip", "maxr", "mom_mag", "PEs_per_hit", "t0", "timeStart", "timeEnd"]
        df_train.drop(columns=col_to_drop, axis=1, inplace=True)

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
