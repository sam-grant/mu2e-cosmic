# Sam Grant 2025
# Training class for ML models with pre-split data

from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import xgboost as xgb
import numpy as np

from pyutils.pylogger import Logger

class Train:
    """Train ML models on pre-split data from assemble_dataset()."""

    def __init__(self, data, model=None, scale_features=True,
                 run="j", out_path=None, verbosity=1):
        """Initialise with assemble_dataset() dict. Default model: XGBClassifier."""
        # Set default model
        if model is None:
            model = xgb.XGBClassifier

        self.model_class = model
        self.scale_features = scale_features

        # Extract data components
        self.X_train = data["X_train"]
        self.X_test = data["X_test"]
        self.y_train = data["y_train"]
        self.y_test = data["y_test"]
        self.metadata_train = data["metadata_train"]
        self.metadata_test = data["metadata_test"]

        # Optional full dataframe
        self.df_full = data.get("df_full", None)

        # Will be set during training
        self.scaler = None
        self.model = None
        self.is_keras = self._is_keras_model()

        # Output directory: output/ml/{run}/results/{tag}
        self.out_path = Path(out_path) if out_path else Path(f"../../output/ml/{run}/results")

        self.logger = Logger(
            print_prefix = "[Train]",
            verbosity = verbosity
        )
        self.logger.log("Initialised", "success")

    def _is_keras_model(self):
        """Check if the model is a Keras model"""
        try:
            import tensorflow as tf
            # Check if it's a compiled Keras model instance
            if isinstance(self.model_class, tf.keras.Model):
                return True
        except ImportError:
            pass
        return False

    def train(self, tag, random_state=42, save_output=False,
              epochs=50, batch_size=32, validation_split=0.0, verbose=1,
              **hyperparams):
        """Train model and return results dict. Pass hyperparams as kwargs."""
        model_type = "Keras" if self.is_keras else self.model_class.__name__
        self.logger.log(
            f"Training model: {model_type}\n"
            f"  Tag: {tag}\n"
            f"  Random state: {random_state}\n"
            f"  Scale features: {self.scale_features}\n",
            "info"
        )

        if self.is_keras:
            self.logger.log(
                f"  Epochs: {epochs}\n"
                f"  Batch size: {batch_size}\n"
                f"  Validation split: {validation_split}\n",
                "info"
            )
        else:
            self.logger.log(
                f"  Hyperparams: {hyperparams}\n",
                "info"
            )

        # Scale features if requested
        if self.scale_features:
            X_train_scaled, X_test_scaled = self._scale_features()
        else:
            X_train_scaled = self.X_train.values
            X_test_scaled = self.X_test.values

        # Train model based on type
        if self.is_keras:
            self.model = self._fit_keras_model(
                X_train_scaled, self.y_train.values,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                **hyperparams
            )
        else:
            self.model = self._fit_model(
                X_train_scaled, self.y_train,
                random_state, **hyperparams
            )

        # Predict on test and train sets
        y_pred, y_proba = self._predict(X_test_scaled)
        _, y_proba_train = self._predict(X_train_scaled)

        self.logger.log("Training complete!", "success")

        # Package results
        results = {
            "tag": tag,
            "model": self.model,
            "feature_names": list(self.X_train.columns),
            "y_train": self.y_train,
            "y_test": self.y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "y_proba_train": y_proba_train,
            "scaler": self.scaler,
            "metadata_test": self.metadata_test
        }

        # Save if requested
        if save_output:
            self._save_results(results, tag)

        return results

    def _scale_features(self):
        """Scale features using StandardScaler (fit on train only)."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled

    def _fit_model(self, X_train, y_train, random_state=42, **hyperparams):
        """Fit sklearn-compatible model. Falls back if random_state unsupported."""
        # Try to pass random_state if model supports it
        try:
            model = self.model_class(random_state=random_state, **hyperparams)
        except TypeError:
            # Model doesn't accept random_state parameter
            self.logger.log(
                f"Note: {self.model_class.__name__} doesn't accept random_state parameter",
                "warning"
            )
            model = self.model_class(**hyperparams)

        model.fit(X_train, y_train)
        return model

    def _fit_keras_model(self, X_train, y_train, epochs, batch_size,
                        validation_split, verbose, **keras_params):
        """Fit compiled Keras model instance."""
        # Use the compiled model instance
        model = self.model_class

        # Fit the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            **keras_params
        )

        # Store training history
        model.history_ = history.history

        return model

    def _predict(self, X_test):
        """Return (y_pred, y_proba) for sklearn or Keras models."""
        if self.is_keras:
            # Keras: predict returns probabilities
            y_proba_raw = self.model.predict(X_test, verbose=0)

            # Handle different output shapes
            if y_proba_raw.shape[1] == 1:
                # Binary classification with sigmoid (shape: n_samples, 1)
                y_proba = y_proba_raw.flatten()
            else:
                # Binary classification with softmax (shape: n_samples, 2)
                y_proba = y_proba_raw[:, 1]

            # Convert probabilities to class predictions (threshold at 0.5)
            y_pred = (y_proba >= 0.5).astype(int)

        else:
            # sklearn: predict returns classes, predict_proba returns probabilities
            y_pred = self.model.predict(X_test)

            # Get probabilities for positive class
            proba_all = self.model.predict_proba(X_test)
            pos_idx = list(self.model.classes_).index(1)
            y_proba = proba_all[:, pos_idx]

        return y_pred, y_proba

    def _save_results(self, results, tag):
        """Save results dict to {out_path}/{tag}/results.pkl."""
        tag_path = self.out_path / tag
        tag_path.mkdir(parents=True, exist_ok=True)

        file_path = tag_path / "results.pkl"

        if self.is_keras:
            # Save Keras model separately
            keras_model_path = tag_path / "keras_model.keras"
            results["model"].save(keras_model_path)
            self.logger.log(f"Keras model saved to {keras_model_path}", "success")

            # Save results without the model (it's saved separately)
            results_copy = results.copy()
            results_copy["model"] = None  # Don't pickle the Keras model
            results_copy["keras_model_path"] = str(keras_model_path)
            joblib.dump(results_copy, file_path)
        else:
            # sklearn models can be pickled directly
            joblib.dump(results, file_path)

        self.logger.log(f"Results saved to {file_path}", "success")
