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


class AnaModel:
    """
    Analyse model after training

    Returns:
    - Confusion matrix (TP, FN, FP, TN)
    - ROC AUC (train and test)
    - ROC curve plot
    - Classification report (precision, recall, F1)
    - Accuracy 
    - Threshold optimisation at given efficiency
    - Threshold optimisation plots

    Works with results from Train class.

    Example:
        >>> from train import Train
        >>> from validate import Validate
        >>>
        >>> # Train model
        >>> trainer = Train(data)
        >>> results = trainer.train(tag="xgb_v1")
        >>>
        >>> # Analyse model 
        >>> ana = AnaValidate(results, data)
        >>> ana.print_summary()
        >>> ana.plot_roc(save_path="roc_curve.png")
    """

    def __init__(self, results, verbosity=1):
        """
        Initialise validator with training results

        Args:
            results: Dictionary from Train.train()
            verbosity: pylogger verbosity
        """
        self.model = results["model"]
        self.feature_names = results.get("feature_names", None)
        self.y_test = results["y_test"]
        self.y_train = results["y_train"]
        self.y_pred = results["y_pred"]
        self.y_proba = results["y_proba"]
        self.y_proba_train = results["y_proba_train"]
        self.tag = results["tag"]

#        self.X_train = data["X_train"]
#        self.X_test = data["X_test"]
#        self.y_train = data["y_train"]

        self.logger = Logger(print_prefix="[Validate]", verbosity=verbosity)
        self.logger.log(f"Initialised validator for model: {self.tag}", "success")

        # Will be computed
        self._train_auc = None
        self._test_auc = None

    def confusion_matrix(self, normalise=False):
        """
        Compute confusion matrix

        Args:
            normalise: Normalise by true class counts

        Returns:
            dict: {
                "TP": true positives,
                "FN": false negatives,
                "FP": false positives,
                "TN": true negatives,
                "matrix": 2x2 confusion matrix [[TN, FP], [FN, TP]]
            }
        """
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
        """
        Compute ROC AUC for both train and test sets

        Returns:
            dict: {
                "train_auc": AUC on training set,
                "test_auc": AUC on test set
            }
        """
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
        """
        Generate classification report with key metrics

        Returns:
            dict: {
                "report": formatted classification report string,
                "accuracy": overall accuracy,
                "precision": precision for positive class,
                "recall": recall for positive class,
                "f1": F1 score for positive class
            }
        """
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

    def plot_roc(self, save_path=None, show=True):
        """
        Plot ROC curve for test set

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
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
        plt.grid(alpha=0.5)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            self.logger.log(f"ROC curve saved to {save_path}", "success")

        if show:
            plt.show()


    def plot_feature_importance(self, feature_names=None, save_path=None, show=True):
        """
        Plot and print feature importance

        Supports tree-based models (feature_importances_) and
        linear models (coef_).

        Args:
            feature_names: List of feature names (optional, retrieved from model if not provided)
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            dict: Feature name to importance mapping, or None if unsupported
        """
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
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            self.logger.log(f"Feature importance saved to {save_path}", "success")

        if show:
            plt.show()

        return dict(zip(feature_names, importances))

    def plot_score_distribution(self, save_path=None, show=True):
        """
        Plot model output score distributions for signal and background

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        # Separate scores by true label
        signal_scores = self.y_proba[self.y_test == 1]
        background_scores = self.y_proba[self.y_test == 0]

        # Plot
        plt.figure()
        plt.hist(background_scores, bins=100, alpha=0.6, label='CE mix',
                 density=True, color='blue')
        plt.hist(signal_scores, bins=100, alpha=0.6, label='CRY',
                 density=True, color='red')

        plt.xlabel('Model output')
        plt.ylabel('Normalised events')
        plt.legend(loc='upper left', bbox_to_anchor=(0.1, 1))
        plt.yscale('log')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            self.logger.log(f"Score distribution saved to {save_path}", "success")

        if show:
            plt.show()

    def check_overlap(self):
        pass

    def print_summary(self):
        """
        Print validation summary

        Displays:
        - Confusion matrix
        - ROC AUC
        - Classification report (includes accuracy, precision, recall, F1)
        """
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


    def find_threshold(self, min_eff=0.999, n_thresholds=10000, save_path=None, show=True):
        """
        Find model threshold at a minimum veto efficiency and plot
        efficiency curves.

        Scans thresholds and computes veto efficiency (TPR) and
        signal efficiency (1 - deadtime, i.e. 1 - FPR). Returns
        the threshold closest to the target efficiency from above.

        Args:
            min_eff: Minimum veto efficiency (default: 0.999)
            n_thresholds: Number of thresholds to scan (default: 10000)
            save_path: Path to save the overlay plot (optional)
            show: Whether to display the plot

        Returns:
            dict: {
                "threshold": optimal threshold,
                "veto_efficiency": actual veto efficiency achieved,
                "deadtime": deadtime (FPR) at this threshold,
                "signal_efficiency": 1 - deadtime
            }
        """
        thresholds = np.linspace(0, 1, n_thresholds)

        y_test = np.asarray(self.y_test)
        y_proba = np.asarray(self.y_proba)

        n_signal = (y_test == 1).sum()
        n_background = (y_test == 0).sum()

        # Vectorised computation of veto efficiency and deadtime
        veto_efficiencies = np.empty(n_thresholds)
        deadtime_losses = np.empty(n_thresholds)

        for i, thr in enumerate(thresholds):
            pred = (y_proba > thr).astype(int)

            tp = ((y_test == 1) & (pred == 1)).sum()
            fp = ((y_test == 0) & (pred == 1)).sum()

            veto_efficiencies[i] = tp / n_signal if n_signal > 0 else 0
            deadtime_losses[i] = fp / n_background if n_background > 0 else 0

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
            f"Threshold optimisation (target {min_eff*100:.2f}% veto efficiency):\n"
            f"  Threshold:         {optimal_threshold:.4f}\n"
            f"  Veto efficiency:   {actual_veto_eff*100:.3f}%\n"
            f"  Signal efficiency: {actual_signal_eff*100:.3f}%\n"
            f"  Deadtime:          {actual_deadtime*100:.3f}%",
            "info"
        )

        # Plot overlay of veto efficiency and signal efficiency
        fig, ax = plt.subplots(figsize=(1.2 * 6.4, 4.8))

        ax.plot(thresholds, veto_efficiencies, linewidth=2,
                color='blue', label='Veto efficiency (CRY vetoed)')
        ax.plot(thresholds, signal_efficiencies, linewidth=2,
                color='red', label='Signal efficiency (CE mix passed)')

        ax.axvline(optimal_threshold, color='green', linestyle='--',
                   alpha=0.8, linewidth=1.5,
                   label=f'{min_eff*100:.2f}% efficiency: {optimal_threshold:.3f}')

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Fraction')
        ax.set_ylim([0.97, 1.01])
        ax.set_xlim([0, 0.1])
        ax.legend(loc='best')
        ax.grid(alpha=0.4)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            self.logger.log(f"Threshold plot saved to {save_path}", "success")

        if show:
            plt.show()

        return {
            "threshold": optimal_threshold,
            "veto_efficiency": actual_veto_eff,
            "deadtime": actual_deadtime,
            "signal_efficiency": actual_signal_eff
        }
