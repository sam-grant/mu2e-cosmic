# System tools
import os
import sys
from pathlib import Path

# Python stack
import pandas as pd
import awkward as ak
import joblib

# pyutils
from pyutils.pylogger import Logger

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
    def __init__(self, run="h", verbosity=1):
        self.run = run
        self.base_in_path = Path(f"../../output/ml/{self.run}/data/")
        self.logger = Logger(print_prefix="[LoadML]", verbosity=verbosity)
        self.logger.log("Initialised", "success")

    def get_results(self):
        """ Raw full processing output from pkl files
        Can use be used seperately for validation
        without cluttering main ML notebook """ 
        # Define input paths
        cry_in_path = self.base_in_path / "CRY_onspill-LH_aw"
        ce_mix_in_path = self.base_in_path/ "CE_mix2BB_onspill-LH_aw"

        # Load data
        cry_data = Load(in_path=cry_in_path).load_pkl()
        ce_mix_data = Load(in_path=ce_mix_in_path).load_pkl()

        self.logger.log("Got full results", "success")

        return cry_data, ce_mix_data

    # def load_model(self, tag, in_path=None):
        # """
        # Load trained model results saved by Train._save_results()

        # Loads the results.pkl file for the given tag, and for Keras
        # models also loads the separately saved .keras file.

        # Args:
            # tag: Model identifier (directory name under in_path)
            # in_path: Base directory containing saved models
                       # (default: ../../output/ml/{run}/models)

        # Returns:
            # dict: Results dictionary matching Train.train() output
        # """
        # if in_path is None:
            # in_path = Path(f"../../output/ml/{self.run}/models")
        # else:
            # in_path = Path(in_path)

        # model_path = in_path / tag
        # results_file = model_path / "results.pkl"

        # if not results_file.exists():
            # self.logger.log(f"Results file not found: {results_file}", "error")
            # return None

        # results = joblib.load(results_file)

        # # Reload Keras model if saved separately
        # if results.get("model") is None and "keras_model_path" in results:
            # keras_path = Path(results["keras_model_path"])
            # if keras_path.exists():
                # import tensorflow as tf
                # results["model"] = tf.keras.models.load_model(keras_path)
                # self.logger.log(f"Loaded Keras model from {keras_path}", "info")
            # else:
                # self.logger.log(f"Keras model file not found: {keras_path}", "error")
                # return None

        # self.logger.log(f"Loaded model '{tag}' from {model_path}", "success")

        # return results

