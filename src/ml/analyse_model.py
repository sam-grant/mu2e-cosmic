# Sam Grant 2025
# Analyse ML model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
from pathlib import Path

from pyutils.pylogger import Logger
from pyutils.pyplot import Plot


class AnaModel:
    """Analyse trained model. Works with results dict from Train.train()."""

    def __init__(self, results, run="j", img_out_path=None, verbosity=1):
        """Initialise with Train.train() results dict."""
        self.model = results["model"]
        self.scaler = results.get("scaler", None)
        self.feature_names = results.get("feature_names", None)
        self.y_test = results["y_test"]
        self.y_train = results["y_train"]
        self.y_pred = results["y_pred"]
        self.y_proba = results["y_proba"]
        self.y_proba_train = results["y_proba_train"]
        self.metadata_test = results.get("metadata_test", None)
        self.tag = results["tag"]

        # Image output directory: output/images/ml/{run}/{tag}/
        if img_out_path is not None:
            self.img_out_path = Path(img_out_path)
        else:
            self.img_out_path = Path(f"../../output/images/ml/{run}/{self.tag}")

        self.logger = Logger(print_prefix="[AnaModel]", verbosity=verbosity)
        self.logger.log(f"Initialised analyser for model: {self.tag}", "success")

        # Will be computed
        self._train_auc = None
        self._test_auc = None

        # Just for styles
        plotter = Plot()

    def _save_fig(self, out_path, default_name):
        """Save current figure, using default filename if no path given"""
        if out_path is None:
            out_path = self.img_out_path / default_name
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        self.logger.log(f"Saved to {out_path}", "success")

    def confusion_matrix(self, normalise=False):
        """Compute confusion matrix. Returns dict with TP, FN, FP, TN, matrix."""
        cm = confusion_matrix(self.y_test, self.y_pred, normalize='true' if normalise else None)

        # Extract components
        tn, fp, fn, tp = cm.ravel()

        result = {
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "matrix": cm
        }

        self.logger.log(
            f"Confusion Matrix:\n"
            f"  TP: {tp:.0f}, FN: {fn:.0f}\n"
            f"  FP: {fp:.0f}, TN: {tn:.0f}",
            "info"
        )

        return result

    def roc_auc(self):
        """Compute ROC AUC for train and test sets."""
        # Test AUC 
        self._test_auc = roc_auc_score(self.y_test, self.y_proba)

        # Train AUC (need to compute probabilities)
        self._train_auc = roc_auc_score(self.y_train, self.y_proba_train)

        self.logger.log(
            f"ROC AUC:\n"
            f"  Train: {self._train_auc:.4f}\n"
            f"  Test:  {self._test_auc:.4f}",
            "info"
        )

        return {
            "train_auc": self._train_auc,
            "test_auc": self._test_auc
        }

    def classification_report(self):
        """Classification report with precision, recall, F1, accuracy."""
        report_str = classification_report(self.y_test, self.y_pred,
                                           target_names=["Background (0)", "Signal (1)"])

        # Also compute key metrics for programmatic access
        report_dict = classification_report(self.y_test, self.y_pred,
                                            target_names=["Background (0)", "Signal (1)"],
                                            output_dict=True)

        result = {
            "report": report_str,
            "accuracy": report_dict["accuracy"],
            "precision": report_dict["Signal (1)"]["precision"],
            "recall": report_dict["Signal (1)"]["recall"],
            "f1": report_dict["Signal (1)"]["f1-score"]
        }

        self.logger.log(f"Classification Report:\n{report_str}", "info")

        return result

    def plot_roc(self, out_path=None, show=True):
        """Plot ROC curve for test set."""
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)

        # Compute AUC if not already done
        if self._test_auc is None:
            self._test_auc = roc_auc_score(self.y_test, self.y_proba)

        # Plot
        plt.figure()
        plt.plot(fpr, tpr, linewidth=2.5, label=f'ROC (AUC = {self._test_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.4)
        plt.tight_layout()

        self._save_fig(out_path, "roc_curve.png")

        if show:
            plt.show()


    def plot_feature_importance(self, feature_names=None, out_path=None, show=True):
        """Plot feature importance (tree-based or linear models)."""
        # Get feature names: argument > results > model attribute
        if feature_names is None:
            if self.feature_names is not None:
                feature_names = self.feature_names
            elif hasattr(self.model, 'feature_names_in_'):
                feature_names = list(self.model.feature_names_in_)
            else:
                self.logger.log(
                    "No feature_names available (not in results or model)",
                    "warning"
                )
                return None

        # Determine importance type
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models (XGBoost, RandomForest, etc.)
            importances = self.model.feature_importances_
            importance_type = "importance"
        elif hasattr(self.model, 'coef_'):
            # Linear models (LogisticRegression, etc.)
            coef = self.model.coef_
            # Handle shape: (1, n_features) for binary classification
            if coef.ndim > 1:
                coef = coef[0]
            importances = np.abs(coef)
            importance_type = "coefficient magnitude"
        else:
            self.logger.log(
                "Model has neither feature_importances_ nor coef_ attribute",
                "warning"
            )
            return None

        # Sort by importance
        indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        # Print feature importance
        print(f"\nFeature {importance_type}:")
        for feat, imp in zip(sorted_names, sorted_importances):
            print(f"  {feat:15s}: {imp:.4f}")

        # Plot
        plt.figure()
        plt.barh(range(len(sorted_names)), sorted_importances[::-1], height=0.8)
        plt.yticks(range(len(sorted_names)), sorted_names[::-1])
        plt.xlabel(f"Feature {importance_type}")
        plt.grid(alpha=0.4)
        plt.tight_layout()

        self._save_fig(out_path, "bar_feature_importance.png")

        if show:
            plt.show()

        return dict(zip(feature_names, importances))

    def find_threshold(self, min_eff=0.999, n_thresholds=10000, out_path=None, show=True):
        """Find highest threshold meeting min_eff event-level veto efficiency."""
        if self.metadata_test is None:
            self.logger.log(
                "No metadata_test available — cannot do event-level threshold scan",
                "error"
            )
            return None

        thresholds = np.linspace(0, 1, n_thresholds)

        y_test = np.asarray(self.y_test)
        y_proba = np.asarray(self.y_proba)
        metadata = self.metadata_test

        # Pre-compute event-level quantities with a single groupby
        df = metadata[["subrun", "event"]].copy()
        df["label"] = y_test
        df["proba"] = y_proba

        event_df = df.groupby(["subrun", "event"]).agg(
            max_proba=("proba", "max"),
            label=("label", "first")
        )

        event_labels = event_df["label"].values
        event_max_proba = event_df["max_proba"].values

        is_signal = (event_labels == 1)
        is_background = (event_labels == 0)
        n_signal_events = is_signal.sum()
        n_background_events = is_background.sum()

        # Vectorised threshold scan — no groupby in the loop
        veto_efficiencies = np.empty(n_thresholds)
        deadtime_losses = np.empty(n_thresholds)

        for i, thr in enumerate(thresholds):
            # Event vetoed if its max coincidence score > threshold
            event_vetoed = event_max_proba >= thr

            tp = (is_signal & event_vetoed).sum()
            fp = (is_background & event_vetoed).sum()

            veto_efficiencies[i] = tp / n_signal_events if n_signal_events > 0 else 0
            deadtime_losses[i] = fp / n_background_events if n_background_events > 0 else 0

        signal_efficiencies = 1 - deadtime_losses

        # Find threshold where veto efficiency >= min_eff
        # (highest threshold that still meets the requirement)
        above_target = np.where(veto_efficiencies >= min_eff)[0]
        if len(above_target) > 0:
            optimal_idx = above_target[-1]  # highest threshold still above target
        else:
            # Fall back to closest
            optimal_idx = np.argmin(np.abs(veto_efficiencies - min_eff))
            self.logger.log(
                f"Target efficiency {min_eff*100:.2f}% not achievable, "
                f"using closest: {veto_efficiencies[optimal_idx]*100:.2f}%",
                "warning"
            )

        optimal_threshold = thresholds[optimal_idx]
        actual_veto_eff = veto_efficiencies[optimal_idx]
        actual_deadtime = deadtime_losses[optimal_idx]
        actual_signal_eff = signal_efficiencies[optimal_idx]

        self.logger.log(
            f"Event-level threshold (target {min_eff*100:.2f}% veto efficiency):\n"
            f"  Threshold:         {optimal_threshold:.4f}\n"
            f"  Veto efficiency:   {actual_veto_eff*100:.3f}%\n"
            f"  Signal efficiency: {actual_signal_eff*100:.3f}%\n"
            f"  Deadtime:          {actual_deadtime*100:.3f}%",
            "info"
        )

        # Plot overlay of veto efficiency and signal efficiency
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

        ax.plot(thresholds, veto_efficiencies, linewidth=2,
                color='blue', label='Veto efficiency (CRY vetoed)')
        ax.plot(thresholds, signal_efficiencies, linewidth=2,
                color='red', label=r'$1-$'+'deadtime (CE mix passed)')

        ax.axvline(optimal_threshold, color='grey', linestyle='--',
                   alpha=0.8, linewidth=1.5,
                   label=f'{min_eff*100:.2f}% efficiency: {optimal_threshold:.4f}')

        # Mark operating point on both curves
        ax.plot(optimal_threshold, actual_veto_eff, 'o', color='blue',
                markersize=8, zorder=5)
        ax.plot(optimal_threshold, actual_signal_eff, 'o', color='red',
                markersize=8, zorder=5)

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Fraction')
        ax.set_ylim([0.96, 1.01])
        ax.set_xlim([0, 0.10])
        ax.legend(loc='best')
        ax.grid(alpha=0.4)

        plt.tight_layout()

        self._save_fig(out_path, "threshold_overlay.png")

        if show:
            plt.show()

        return {
            "threshold": optimal_threshold,
            "veto_efficiency": actual_veto_eff,
            "deadtime": actual_deadtime,
            "signal_efficiency": actual_signal_eff
        }

    def plot_score_distribution(self, threshold, out_path=None, show=True,
                                nbins=100):
        """Score distributions for signal/background. Full range and zoomed to 2*threshold."""
        # Separate scores by true label
        signal_scores = self.y_proba[self.y_test == 1]
        background_scores = self.y_proba[self.y_test == 0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*6.4, 4.8))

        # Left: full range
        for ax, xrange in [(ax1, (0, 1)), (ax2, (0, 2 * threshold))]:
            ax.hist(background_scores, bins=nbins, range=xrange,
                    alpha=0.6, label='CE mix',
                    density=True, color='blue')
            ax.hist(signal_scores, bins=nbins, range=xrange,
                    alpha=0.6, label='CRY',
                    density=True, color='red')
            ax.axvline(threshold, color='grey', linestyle='--', alpha=0.8,
                       linewidth=1.5, label=f'Threshold: {threshold:.4f}')
            ax.set_xlabel('Model output')
            ax.set_ylabel('Normalised events')
            ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1))
            ax.set_yscale('log')
            ax.grid(alpha=0.4)

        plt.tight_layout()

        self._save_fig(out_path, "h1_score_distribution.png")

        if show:
            plt.show()

    def plot_physics_by_score(self, df_full, threshold, above=False, variables=None,
                               nbins=50, out_path=None, show=True):
        """Plot feature distributions for events above or below threshold, split by CRY/CE mix."""
        if self.metadata_test is None:
            self.logger.log("No metadata_test available — cannot match events", "error")
            return

        # Match test events back to full DataFrame
        df_test = df_full.loc[self.metadata_test.index].copy()
        df_test["score"] = np.asarray(self.y_proba)
        df_test["label"] = np.asarray(self.y_test)

        # Select events above or below threshold
        if above:
            df_sel = df_test[df_test["score"] >= threshold]
            score_label = f"score >= {threshold:.4f}"
            default_fname = "h1_high_score_physics.png"
        else:
            df_sel = df_test[df_test["score"] < threshold]
            score_label = f"score < {threshold:.4f}"
            default_fname = "h1_low_score_physics.png"

        # Print selected CRY event metadata
        df_sel_cry = df_sel[df_sel["label"] == 1]
        self.logger.log(
            f"Selected events ({score_label}):\n"
            f"  CRY:    {len(df_sel_cry)}\n"
            f"  CE mix: {(df_sel['label'] == 0).sum()}",
            "info"
        )
        if not above and len(df_sel_cry) > 0:
            meta_cols = ["event", "subrun", "score"]
            meta_cols = [c for c in meta_cols if c in df_sel_cry.columns]
            print(f"\nLow-score CRY events:\n{df_sel_cry[meta_cols].to_string(index=False)}\n")
            # Save to CSV
            csv_path = self.img_out_path / "low_score_cry.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_sel_cry.to_csv(csv_path, index=False)
            self.logger.log(f"Saved low-score CRY events to {csv_path}", "success")

        # Default to trained feature names
        if variables is None:
            if self.feature_names is not None:
                variables = list(self.feature_names)
            else:
                self.logger.log("No feature_names or variables provided", "warning")
                return
        # Filter to columns that actually exist
        variables = [v for v in variables if v in df_sel.columns]

        if len(variables) == 0:
            self.logger.log("No matching columns found in DataFrame", "warning")
            return

        # Split by label
        df_cry = df_sel[df_sel["label"] == 1]
        df_mix = df_sel[df_sel["label"] == 0]

        # Plot grid
        ncols = 3
        nrows = (len(variables) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.8 * nrows))
        axes = np.atleast_2d(axes).flatten()

        for i, var in enumerate(variables):
            ax = axes[i]
            mix_vals = df_mix[var].dropna()
            cry_vals = df_cry[var].dropna()

            # Common range
            all_vals = pd.concat([mix_vals, cry_vals])
            lo, hi = all_vals.quantile(0.01), all_vals.quantile(0.99)

            ax.hist(mix_vals, bins=nbins, range=(lo, hi),
                    alpha=0.6, label="CE mix", density=True, color="blue")
            ax.hist(cry_vals, bins=nbins, range=(lo, hi),
                    alpha=0.6, label="CRY", density=True, color="red")
            ax.set_xlabel(var)
            ax.legend()
            ax.grid(alpha=0.4)

        # Hide unused axes
        for i in range(len(variables), len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f"Feature distributions ({score_label})", y=1.01)
        plt.tight_layout()

        self._save_fig(out_path, default_fname)

        if show:
            plt.show()

    @staticmethod
    def event_level_confusion(y_pred, y_true, metadata):
        """Event-level confusion matrix. Vetoes event if ANY coincidence is flagged."""
        df = metadata[["subrun", "event"]].copy()
        df["pred"] = np.asarray(y_pred).astype(int)
        df["label"] = np.asarray(y_true).astype(int)

        # Group by event: veto if ANY coincidence flagged, label from first row
        event_df = df.groupby(["subrun", "event"]).agg(
            vetoed=("pred", "max"),
            label=("label", "first")
        ).reset_index()

        tp = ((event_df["label"] == 1) & (event_df["vetoed"] == 1)).sum()
        tn = ((event_df["label"] == 0) & (event_df["vetoed"] == 0)).sum()
        fp = ((event_df["label"] == 0) & (event_df["vetoed"] == 1)).sum()
        fn = ((event_df["label"] == 1) & (event_df["vetoed"] == 0)).sum()

        return {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "n_events": len(event_df),
        }

    def money_table(self, X, y, metadata, threshold=None,
                    dT_min=0, dT_max=150, save_csv=True):
        """Event-level comparison of ML model vs dT window cut."""
        if self.feature_names is None:
            self.logger.log("No feature_names available", "error")
            return None

        # --- Score coincidences with the ML model ---
        X_scaled = X[self.feature_names].values
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_scaled)

        # Get model probabilities
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_scaled)
            pos_idx = list(self.model.classes_).index(1)
            y_proba = proba[:, pos_idx]
        else:
            # Keras or similar
            y_proba = self.model.predict(X_scaled, verbose=0).flatten()

        if threshold is None:
            threshold = 0.5
        y_pred_ml = (y_proba >= threshold).astype(int)

        # --- Simple dT cut: same logic as check_dT_window_results ---
        dT = X["dT"].values
        y_pred_dt = (
            (~np.isnan(dT))
            & (dT >= dT_min)
            & (dT <= dT_max)
        ).astype(int)

        # --- Labels ---
        y_true = np.asarray(y)

        # --- Aggregate to event level ---
        cm_ml = self.event_level_confusion(y_pred_ml, y_true, metadata)
        tp, tn, fp, fn = cm_ml["tp"], cm_ml["tn"], cm_ml["fp"], cm_ml["fn"]

        cm_dt = self.event_level_confusion(y_pred_dt, y_true, metadata)
        tp_dt, tn_dt = cm_dt["tp"], cm_dt["tn"]
        fp_dt, fn_dt = cm_dt["fp"], cm_dt["fn"]

        # --- Compute metrics for both ---
        veto_eff = tp / (tp + fn) if (tp + fn) > 0 else 0
        deadtime = fp / (tn + fp) if (tn + fp) > 0 else 0
        veto_purity = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fom = veto_eff * (1 - deadtime)

        veto_eff_dt = tp_dt / (tp_dt + fn_dt) if (tp_dt + fn_dt) > 0 else 0
        deadtime_dt = fp_dt / (tn_dt + fp_dt) if (tn_dt + fp_dt) > 0 else 0
        veto_purity_dt = tp_dt / (tp_dt + fp_dt) if (tp_dt + fp_dt) > 0 else 0
        accuracy_dt = (tp_dt + tn_dt) / (tp_dt + tn_dt + fp_dt + fn_dt)
        fom_dt = veto_eff_dt * (1 - deadtime_dt)

        # --- Metrics comparison table ---
        metrics_df = pd.DataFrame({
            "Metric": [
                "Veto efficiency",
                "Deadtime",
                "Veto purity",
                "Overall accuracy",
                "Figure of merit",
            ],
            "ML model": [
                f"{veto_eff*100:.3f}%",
                f"{deadtime*100:.3f}%",
                f"{veto_purity*100:.3f}%",
                f"{accuracy*100:.3f}%",
                f"{fom*100:.3f}%",
            ],
            f"dT cut [{dT_min}, {dT_max}] ns": [
                f"{veto_eff_dt*100:.3f}%",
                f"{deadtime_dt*100:.3f}%",
                f"{veto_purity_dt*100:.3f}%",
                f"{accuracy_dt*100:.3f}%",
                f"{fom_dt*100:.3f}%",
            ],
            "Difference": [
                f"{(veto_eff - veto_eff_dt)*100:+.3f}%",
                f"{(deadtime - deadtime_dt)*100:+.3f}%",
                f"{(veto_purity - veto_purity_dt)*100:+.3f}%",
                f"{(accuracy - accuracy_dt)*100:+.3f}%",
                f"{(fom - fom_dt)*100:+.3f}%",
            ],
            "Description": [
                "Fraction of cosmics vetoed",
                "Fraction of CE mix vetoed",
                "Of vetoed events, fraction that are cosmics",
                "Overall correct classification rate",
                "eff_veto * (1 - deadtime)",
            ],
        })

        # --- Confusion matrix comparison table ---
        confusion_df = pd.DataFrame({
            "Category": [
                "True Positives (CRY vetoed)",
                "True Negatives (CE mix passed)",
                "False Positives (CE mix vetoed)",
                "False Negatives (CRY passed)",
            ],
            "ML model": [tp, tn, fp, fn],
            "dT cut": [tp_dt, tn_dt, fp_dt, fn_dt],
            "Difference": [tp - tp_dt, tn - tn_dt, fp - fp_dt, fn - fn_dt],
        })

        # --- Save to CSV ---
        if save_csv:
            self.img_out_path.mkdir(parents=True, exist_ok=True)

            metrics_path = self.img_out_path / "metrics_comparison.csv"
            metrics_df.to_csv(metrics_path, index=False)

            confusion_path = self.img_out_path / "confusion_comparison.csv"
            confusion_df.to_csv(confusion_path, index=False)

            self.logger.log(
                f"Saved comparison tables to {self.img_out_path}",
                "success"
            )

        return {
            "metrics": metrics_df,
            "confusion": confusion_df,
        }

    def print_summary(self):
        """Print confusion matrix, ROC AUC, and classification report."""
        print(f"\n{'='*60}")
        print(f"  VALIDATION SUMMARY: {self.tag}")
        print(f"{'='*60}\n")

        # Confusion matrix
        self.confusion_matrix()

        # ROC AUC
        self.roc_auc()

        # Classification report (includes accuracy metrics)
        self.classification_report()

        print(f"{'='*60}\n")
