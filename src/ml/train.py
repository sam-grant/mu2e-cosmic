from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from pyutils.pylogger import Logger

class Train():
    def __init__(self, data):
        self.data = data
        self.logger = Logger("[Train]", verbosity=1)

    def scale():
        """ Normalise/scale the data 
        (not strictly needed for BDTs)"""
        scalar = StandardScalar()
        self.data["X_train_scaled"] = scalar.fit_transform(self.data["X_train"])
        self.data["X_test_scaled"] = scalar.fit_transform(self.data["X_test"])
        self.logger.log("Scaling complete!", "success")
        return self.data

    def train_xgboost(n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=42):
        """"""
        # Scale (no harm done)
        data = self.scale()

        # Configure the model
        model = xgb.XGBClassifier(
            n_estimators=n_estimators
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )

        # Fit the model to the scaled training data
        model.fit(data["X_train_scaled"], data["y_train"])

        self.logger.log("Training complete!", "success")
        return model