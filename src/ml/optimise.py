# Sam Grant 2025
# Hyperparameter optimisation for ML models

import os
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parents[1]

from pyutils.pylogger import Logger
from train import Train
from validate import Validate


class Optimise:
    """Grid search over Train -> Validate pipeline. Minimises deadtime at target veto efficiency."""

    def __init__(self, data, run="j", min_efficiency=0.999,
                 model=None, scale_features=True, verbosity=1):
        """Initialise with assemble_dataset() dict."""
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
        """Run grid search. Returns best result (lowest deadtime meeting efficiency constraint)."""
        # Build all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        self.logger.log(
            f"Grid search: {len(combinations)} combinations "
            f"over {param_names}",
            "info"
        )

        best_deadtime = np.inf

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
            ana = Validate(training_results, run=self.run, verbosity=0)
            ana.roc_auc()
            threshold_results = ana.find_threshold(
                min_eff=self.min_efficiency,
                plot=False,
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

            # Track best: lowest deadtime among combos meeting efficiency constraint
            meets_constraint = threshold_results["veto_efficiency"] >= self.min_efficiency
            if meets_constraint and threshold_results["deadtime"] < best_deadtime:
                best_deadtime = threshold_results["deadtime"]
                self.best_result = {
                    "training_results": training_results,
                    "threshold_results": threshold_results,
                    "hyperparams": hyperparams,
                    "tag": tag,
                }

        if self.best_result is not None:
            self.logger.log(
                f"Grid search complete. Best: {self.best_result['tag']} "
                f"(deadtime: {best_deadtime*100:.3f}%, "
                f"veto eff: {self.best_result['threshold_results']['veto_efficiency']*100:.3f}%)",
                "success"
            )
            # Plot threshold overlay for winning scan only
            best_ana = Validate(
                self.best_result["training_results"],
                run=self.run, verbosity=0
            )
            best_ana.find_threshold(
                min_eff=self.min_efficiency,
                plot=True,
                show=True
            )
        else:
            self.logger.log(
                f"Grid search complete. No combination met "
                f"{self.min_efficiency*100:.2f}% veto efficiency constraint",
                "warning"
            )

        return self.best_result

    def grid_search_cv(self, param_grid, n_folds=5, tag_prefix="scan_cv",
                       random_state=42):
        """K-fold CV grid search. Averages metrics across folds per hyperparam combo."""
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        # CV folds over train set only, val/test held out
        X = self.data["X_train"].reset_index(drop=True)
        y = self.data["y_train"].reset_index(drop=True)
        metadata = self.data["metadata_train"].reset_index(drop=True)

        # Event-level groups for fold splitting
        groups = metadata["subrun"].astype(str) + "_" + metadata["event"].astype(str)

        gkf = GroupKFold(n_splits=n_folds)
        folds = list(gkf.split(X, y, groups=groups))

        self.logger.log(
            f"CV grid search: {len(combinations)} combinations x {n_folds} folds "
            f"= {len(combinations) * n_folds} fits over {param_names}",
            "info"
        )

        best_deadtime = np.inf

        for i, combo in enumerate(combinations):
            hyperparams = dict(zip(param_names, combo))
            tag = f"{tag_prefix}_{i:03d}"

            fold_results = []

            for k, (train_idx, test_idx) in enumerate(folds):
                # Build fold data dict
                fold_data = {
                    "X_train": X.iloc[train_idx],
                    "X_test": X.iloc[test_idx],
                    "y_train": y.iloc[train_idx],
                    "y_test": y.iloc[test_idx],
                    "metadata_train": metadata.iloc[train_idx],
                    "metadata_test": metadata.iloc[test_idx],
                }

                trainer = Train(
                    fold_data,
                    model=self.model,
                    scale_features=self.scale_features,
                    run=self.run,
                    verbosity=0
                )
                training_results = trainer.train(
                    tag=f"{tag}_fold{k}",
                    random_state=random_state,
                    save_output=False,
                    **hyperparams
                )

                ana = Validate(training_results, run=self.run, verbosity=0)
                ana.roc_auc()
                threshold_results = ana.find_threshold(
                    min_eff=self.min_efficiency,
                    plot=False,
                    show=False
                )

                fold_results.append({
                    "auc": ana._test_auc,
                    "threshold": threshold_results["threshold"],
                    "veto_efficiency": threshold_results["veto_efficiency"],
                    "deadtime": threshold_results["deadtime"],
                    "signal_efficiency": threshold_results["signal_efficiency"],
                })

            # Average across folds
            mean_auc = np.mean([r["auc"] for r in fold_results])
            mean_deadtime = np.mean([r["deadtime"] for r in fold_results])
            mean_veto_eff = np.mean([r["veto_efficiency"] for r in fold_results])
            mean_signal_eff = np.mean([r["signal_efficiency"] for r in fold_results])
            mean_threshold = np.mean([r["threshold"] for r in fold_results])
            std_deadtime = np.std([r["deadtime"] for r in fold_results])
            std_veto_eff = np.std([r["veto_efficiency"] for r in fold_results])

            row = {
                "tag": tag,
                **hyperparams,
                "auc": mean_auc,
                "threshold": mean_threshold,
                "veto_efficiency": mean_veto_eff,
                "veto_efficiency_std": std_veto_eff,
                "deadtime": mean_deadtime,
                "deadtime_std": std_deadtime,
                "signal_efficiency": mean_signal_eff,
            }
            self.results_table.append(row)

            self.logger.log(
                f"[{i+1}/{len(combinations)}] {tag}: "
                f"deadtime={mean_deadtime*100:.3f}±{std_deadtime*100:.3f}%, "
                f"veto_eff={mean_veto_eff*100:.3f}±{std_veto_eff*100:.3f}%",
                "info"
            )

            # Track best: lowest mean deadtime meeting mean efficiency constraint
            meets_constraint = mean_veto_eff >= self.min_efficiency
            if meets_constraint and mean_deadtime < best_deadtime:
                best_deadtime = mean_deadtime
                self.best_result = {
                    "hyperparams": hyperparams,
                    "tag": tag,
                    "mean_deadtime": mean_deadtime,
                    "mean_veto_efficiency": mean_veto_eff,
                    "fold_results": fold_results,
                }

        if self.best_result is not None:
            self.logger.log(
                f"CV grid search complete. Best: {self.best_result['tag']} "
                f"(deadtime: {best_deadtime*100:.3f}%, "
                f"veto eff: {self.best_result['mean_veto_efficiency']*100:.3f}%)",
                "success"
            )
        else:
            self.logger.log(
                f"CV grid search complete. No combination met "
                f"{self.min_efficiency*100:.2f}% mean veto efficiency constraint",
                "warning"
            )

        return self.best_result

    def get_summary(self):
        """Results as DataFrame, sorted by deadtime ascending."""
        if not self.results_table:
            self.logger.log("No results yet — run grid_search first", "warning")
            return None

        df = pd.DataFrame(self.results_table)
        df = df.sort_values("deadtime", ascending=True).reset_index(drop=True)
        return df

    def print_summary(self):
        """Print results summary table."""
        df = self.get_summary()
        if df is not None:
            print(f"\n{'='*80}")
            print(f"Optimisation summary ({len(df)} runs, "
                  f"min efficiency = {self.min_efficiency*100:.1f}%)")
            print(f"{'='*80}")
            print(df.to_string(index=False))
            print(f"{'='*80}\n")

    def save_summary(self, out_path=None):
        """Save results summary to CSV."""
        df = self.get_summary()
        if df is None:
            return

        if out_path is None:
            out_path = REPO_ROOT / f"output/ml/{self.run}/results/optimisation_summary.csv"

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        self.logger.log(f"Summary saved to {out_path}", "success")
