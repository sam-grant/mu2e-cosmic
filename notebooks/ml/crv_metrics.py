

####
####
###

class CrvMetrics:
    """ Get CRV metrics from model
    
    This needs to work with event level metrics
    so we need to group things by event and subrun 

    Also need to be able to handle just straightforward dT cut 

    
    """
    def __init__(self, model, X_test, y_test, metadata, threshold):
        """
        """
        import pandas as pd
        
        self.model = model 
        self.X_test = X_test
        self.y_test = y_test
        self.metadata = metadata 
        self.pred = model.predict_proba(self.X_test)[:, 1]
        self.y_pred = (self.pred >= threshold).astype(int)
        
        # Merge metadata with test data
        self.df_test = pd.concat([
            self.metadata.reset_index(drop=True),
            self.X_test.reset_index(drop=True),
            pd.DataFrame({'label': self.y_test.values, 'score': self.pred, 'veto': self.y_pred})
        ], axis=1)

        print(f"Number of test events: {len(self.y_test)}")


    def cm_from_model(self):
        # from sklearn.metrics import confusion_matrix
        # return confusion_matrix(self.y_true, self.y_pred)
            # Calculate confusion matrix elements
        tp = ((self.y_test == 1) & (self.y_pred == 1)).sum()
        fn = ((self.y_test == 1) & (self.y_pred == 0)).sum()
        fp = ((self.y_test == 0) & (self.y_pred == 1)).sum()
        tn = ((self.y_test == 0) & (self.y_pred == 0)).sum()
        print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")

        # Don't we also want metadata? 
        # Kind of want to know which events where dealing with
        return tp, fn, fp, tn

    def cm_from_cut(self, dt_window=[-25, 150]):
        """ Confusion matrix from simple dT cut """
        # Merge metadata and y_test
        df = self.metadata.copy()
        df["y_test"] = self.y_test.values
        df["dT"] = self.X_test["dT"].values
        df["event"] = self.metadata["event"].values
        df["subrun"] = self.metadata["subrun"].values

        # Apply dT cut
        # Veto if dT is valid and in window
        veto_condition = (~np.isnan(df["dT"])) & (df["dT"] >= dt_window[0]) & (df["dT"] <= dt_window[1])
        df["vetoed"] = veto_condition

        # Group by event and subrun to get event level veto
        event_veto = df.groupby(['subrun', 'event'])['vetoed'].max().reset_index() 

        print(event_veto)
        print(df)

        # total = len(event_veto)
        # vetoed = event_veto['vetoed'].sum()
        # unvetoed = total - vetoed

        # results = pd.DataFrame({
        #     'Metric': ['Total', 'Vetoed', 'Unvetoed', 'Efficiency'],
        #     'Value': [total, vetoed, unvetoed, f'{100*vetoed/total:.2f}%'] 
        # })

        # df_event_veto = df.groupby(['subrun', 'event'])['dT']

        # df["y_pred"] = veto_condition.astype(int)

        # # Calculate confusion matrix elements
        # tp = ((df["y_test"] == 1) & (df["y_pred"] == 1)).sum()
        # fn = ((df["y_test"] == 1) & (df["y_pred"] == 0)).sum()
        # fp = ((df["y_test"] == 0) & (df["y_pred"] == 1)).sum()
        # tn = ((df["y_test"] == 0) & (df["y_pred"] == 0)).sum()
        # print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
        # return tp, fn, fp, tn


# System tools  
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore") # suppress warnings

# Python stack 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak

# ML tools
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

# Internal modules 
sys.path.extend(["../../src/core", "../../src/utils"])
from draw import Draw
from io_manager import Load

# pyutils 
from pyutils.pyplot import Plot
plotter = Plot() # just use this for styles

# def metrics_from_cut(X, dt_window=[-25, 150]):

#     dt = X["dT"].values
#     dt_min, dt_max = dt_window

#     # Veto if dT is valid and in window
#     veto_condition = (~np.isnan(dt)) & (dt >= dt_min) & (dt <= dt_max)

#     # Event is vetoed if ANY coincidence passes cut
#     df_cry_with_veto = df_cry.copy()
#     df_cry_with_veto['vetoed'] = veto_condition

#     event_veto = df_cry_with_veto.groupby(['subrun', 'event'])['vetoed'].max().reset_index()

#     total = len(event_veto)
#     vetoed = event_veto['vetoed'].sum()
#     unvetoed = total - vetoed

#     results = pd.DataFrame({
#         'Metric': ['Total', 'Vetoed', 'Unvetoed', 'Efficiency'],
#         'Value': [total, vetoed, unvetoed, f'{100*vetoed/total:.2f}%'] 
#     })


def main():
    # Load
    run = "g"
    base_in_path = Path(f"../../output/ml/{run}/data/")
    cry_in_path = base_in_path / "CRY_onspill-LH_aw"
    ce_mix_in_path = base_in_path/ "CE_mix2BB_onspill-LH_aw"

    cry_data = Load(in_path=cry_in_path).load_pkl()
    ce_mix_data = Load(in_path=ce_mix_in_path).load_pkl()

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
    col_to_drop = ["d0", "tanDip", "maxr", "mom_mag", "PEs_per_hit", "t0", 
                "timeStart", "timeEnd"]
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

    # Load the model
    model = xgb.XGBClassifier()
    model.load_model("../../output/ml/g/models/trained_xgboost.json")

    crv_metrics = CrvMetrics(
        model=model,
        X_test=X_test,
        y_test=y_test,
        metadata=metadata_test,
        threshold=0.5
    )

    cm = crv_metrics.cm_from_cut()
    # print(cm)

if __name__ == "__main__":
    main()

