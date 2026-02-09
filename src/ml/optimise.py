# Sam Grant 2025
# Hyperparameter optimisation for ML models

import itertools
import numpy as np
import pandas as pd
from pathlib import Path

from pyutils.pylogger import Logger
from train import Train
from analyse_model import AnaModel


class Optimise:
    """
    Hyperparameter optimisation over the Train → AnaModel pipeline

    Runs grid search over a parameter space, training and evaluating
    each combination. Ranks results by veto efficiency at a given
    minimum efficiency constraint.

    Example:
        >>> data = assembler.assemble_dataset()
        >>> opt = Optimise(data, run="j")
        >>> param_grid = {
        ...     "n_estimators": [100, 200, 500],
        ...     "max_depth": [3, 5, 7],
        ...     "learning_rate": [0.01, 0.1, 0.3]
        ... }
        >>> results = opt.grid_search(param_grid, tag_prefix="xgb_scan")
        >>> opt.print_summary()
    """

    def __init__(self, data, run="j", min_efficiency=0.99,
                 model=None, scale_features=True, verbosity=1):
        """
        Initialise optimiser

        Args:
            data: Dictionary from assemble_dataset()
            run: Run identifier
            min_efficiency: Minimum veto efficiency constraint for
                            threshold selection (default: 0.99)
            model: Model class (default: XGBClassifier via Train)
            scale_features: Whether to scale features (default: True)
            verbosity: Logger verbosity
        """
        self.data = data
        self.run = run
        self.min_efficiency = min_efficiency
        self.model = model
        self.scale_features = scale_features

        self.results_table = []  # List of dicts for summary DataFrame
        self.best_result = None

        self.logger = Logger(print_prefix="[Optimise]", verbosity=verbosity)
        self.logger.log("Initialised", "success")

    def grid_search(self, param_grid, tag_prefix="scan",
                    save_output=False, random_state=42):
        """
        Run grid search over hyperparameter combinations

        Args:
            param_grid: Dictionary of parameter names → lists of values
                        e.g. {"n_estimators": [100, 200], "max_depth": [3, 5]}
            tag_prefix: Prefix for model tags (each run gets a unique suffix)
            save_output: Whether to save each model's results to disk
            random_state: Random seed for training

        Returns:
            dict: Best result dictionary from Train.train(),
                  augmented with threshold info from AnaModel
        """
        # Build all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        self.logger.log(
            f"Grid search: {len(combinations)} combinations "
            f"over {param_names}",
            "info"
        )

        best_veto_eff = -1

        for i, combo in enumerate(combinations):
            hyperparams = dict(zip(param_names, combo))
            tag = f"{tag_prefix}_{i:03d}"

            self.logger.log(
                f"[{i+1}/{len(combinations)}] {tag}: {hyperparams}",
                "info"
            )

            # Train
            trainer = Train(
                self.data,
                model=self.model,
                scale_features=self.scale_features,
                run=self.run,
                verbosity=0
            )
            training_results = trainer.train(
                tag=tag,
                random_state=random_state,
                save_output=save_output,
                **hyperparams
            )

            # Analyse — find threshold at minimum efficiency
            ana = AnaModel(training_results, run=self.run, verbosity=0)
            threshold_results = ana.find_threshold(
                min_eff=self.min_efficiency,
                show=False
            )

            # Record results
            row = {
                "tag": tag,
                **hyperparams,
                "auc": ana._test_auc,
                "threshold": threshold_results["threshold"],
                "veto_efficiency": threshold_results["veto_efficiency"],
                "signal_efficiency": threshold_results["signal_efficiency"],
                "deadtime": threshold_results["deadtime"],
            }
            self.results_table.append(row)

            # Track best
            if threshold_results["veto_efficiency"] > best_veto_eff:
                best_veto_eff = threshold_results["veto_efficiency"]
                self.best_result = {
                    "training_results": training_results,
                    "threshold_results": threshold_results,
                    "hyperparams": hyperparams,
                    "tag": tag,
                }

        self.logger.log(
            f"Grid search complete. Best: {self.best_result['tag']} "
            f"(veto eff: {best_veto_eff*100:.3f}%)",
            "success"
        )

        return self.best_result

    def get_summary(self):
        """
        Get results summary as a DataFrame, sorted by veto efficiency

        Returns:
            pd.DataFrame: One row per hyperparameter combination
        """
        if not self.results_table:
            self.logger.log("No results yet — run grid_search first", "warning")
            return None

        df = pd.DataFrame(self.results_table)
        df = df.sort_values("veto_efficiency", ascending=False).reset_index(drop=True)
        return df

    def print_summary(self):
        """Print results summary table"""
        df = self.get_summary()
        if df is not None:
            print(f"\n{'='*80}")
            print(f"Optimisation summary ({len(df)} runs, "
                  f"min efficiency = {self.min_efficiency*100:.1f}%)")
            print(f"{'='*80}")
            print(df.to_string(index=False))
            print(f"{'='*80}\n")

    def save_summary(self, out_path=None):
        """
        Save results summary to CSV

        Args:
            out_path: Output file path
                      (default: output/ml/{run}/results/optimisation_summary.csv)
        """
        df = self.get_summary()
        if df is None:
            return

        if out_path is None:
            out_path = Path(f"../../output/ml/{self.run}/results/optimisation_summary.csv")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        self.logger.log(f"Summary saved to {out_path}", "success")
