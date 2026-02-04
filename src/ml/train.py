# Sam Grant 2025
# Training class for ML models with pre-split data

from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import xgboost as xgb
import numpy as np

from pyutils.pylogger import Logger

class Train:
    """
    Train ML models on pre-split data from assemble_dataset()

    Supports XGBoost (default), sklearn-compatible models, and TensorFlow Keras models.
    Handles feature scaling, model training, and result persistence.
    """

    def __init__(self, data, model=None, scale_features=True, verbosity=1):
        """
        Initialise trainer with pre-split data

        Args:
            data: Dictionary from assemble_dataset() with keys:
                - X_train, X_test: Feature DataFrames (unscaled)
                - y_train, y_test: Labels
                - metadata_train, metadata_test: Event/subrun info
                - df_train_full: Full combined DataFrame
            model: Model class or compiled Keras model (default: xgb.XGBClassifier)
                   - For sklearn: pass model class 
                   - For Keras: pass compiled model instance
            scale_features: Whether to apply StandardScaler (default: True)
                    Set to False for tree-based models that don't need scaling
        """
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
        self.df_train_full = data.get("df_train_full", None)

        # Will be set during training
        self.scaler = None
        self.model = None
        self.is_keras = self._is_keras_model()

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
              output_dir="../../output/ml/models",
              epochs=50, batch_size=32, validation_split=0.0, verbose=1,
              **hyperparams):
        """
        Train the model on pre-split data

        Args:
            tag: Identifier for this training run
            random_state: Random seed for reproducibility (default: 42)
            save_output: Save results to disk (default: False)
            output_dir: Directory for saved results (default: "../../output/ml/models")

            # Keras-specific parameters
            epochs: Number of training epochs for Keras (default: 50, ignored for sklearn)
            batch_size: Batch size for Keras (default: 32, ignored for sklearn)
            validation_split: Fraction of training data to use for validation in Keras (default: 0.0)
            verbose: Verbosity level for Keras training (default: 1)

            **hyperparams: Model-specific hyperparameters
                For XGBoost: n_estimators, max_depth, learning_rate, etc.
                For sklearn models: pass model-specific parameters
                For Keras: callbacks, class_weight, etc.

        Returns:
            dict: {
                "tag": tag,
                "model": trained model,
                "y_test": test labels,
                "y_pred": predictions,
                "y_proba": prediction probabilities for positive class,
                "scaler": fitted scaler (if scaling enabled, else None),
                "metadata_test": test metadata
            }
        """
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
            X_train_processed, X_test_processed = self._scale_features()
        else:
            X_train_processed = self.X_train.values
            X_test_processed = self.X_test.values

        # Train model based on type
        if self.is_keras:
            self.model = self._fit_keras_model(
                X_train_processed, self.y_train.values,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                **hyperparams
            )
        else:
            self.model = self._fit_model(
                X_train_processed, self.y_train,
                random_state, **hyperparams
            )

        # Predict on test set
        y_pred, y_proba = self._predict(X_test_processed)

        print(f"Training complete!\n")

        # Package results
        results = {
            "tag": tag,
            "model": self.model,
            "y_test": self.y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "scaler": self.scaler,
            "metadata_test": self.metadata_test
        }

        # Save if requested
        if save_output:
            self._save_results(results, tag, output_dir)

        return results

    def _scale_features(self):
        """
        Scale features using StandardScaler

        Fits scaler on training data only to prevent data leakage.

        Returns:
            tuple: (X_train_scaled, X_test_scaled) as numpy arrays
        """
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled

    def _fit_model(self, X_train, y_train, random_state, **hyperparams):
        """
        Fit sklearn-compatible models with hyperparameters

        Automatically detects if the model accepts random_state parameter.
        If not, trains without it (useful for some sklearn models).

        Args:
            X_train: Training features
            y_train: Training labels
            random_state: Random seed
            **hyperparams: Model-specific hyperparameters

        Returns:
            Trained model instance
        """
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
        """
        Fit Keras model

        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            verbose: Verbosity level
            **keras_params: Additional Keras fit parameters (callbacks, class_weight, etc.)

        Returns:
            Trained Keras model
        """
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
        """
        Get predictions and probabilities for both sklearn and Keras models

        Args:
            X_test: Test features

        Returns:
            tuple: (y_pred, y_proba)
                - y_pred: Class predictions (0 or 1)
                - y_proba: Probabilities for positive class (label=1)
        """
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

    def _save_results(self, results, tag, output_dir):
        """
        Save results to disk using joblib

        Args:
            results: Results dictionary from train()
            tag: Model identifier
            output_dir: Base directory for saved results

        Creates: {output_dir}/{tag}/results.pkl

        Note: For Keras models, the model itself should be saved separately
              using model.save() as Keras models may not pickle well.
        """
        out_path = Path(output_dir) / tag
        out_path.mkdir(parents=True, exist_ok=True)

        # Save results
        file_path = out_path / "results.pkl"

        if self.is_keras:
            # Save Keras model separately
            keras_model_path = out_path / "keras_model.keras"
            results["model"].save(keras_model_path)
            self.logger(f"Keras model saved to {keras_model_path}", "success")

            # Save results without the model (it's saved separately)
            results_copy = results.copy()
            results_copy["model"] = None  # Don't pickle the Keras model
            results_copy["keras_model_path"] = str(keras_model_path)
            joblib.dump(results_copy, file_path)
        else:
            # sklearn models can be pickled directly
            joblib.dump(results, file_path)

        self.logger.log(f"Results saved to {file_path}", "success")
