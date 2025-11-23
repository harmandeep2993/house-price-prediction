import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from src.utils.config_loader import load_config


class ModelTrainer:
    """
    The ModelTrainer class handles the full ML workflow:
    - Load the processed dataset
    - Split into train/test sets
    - Train a Random Forest model
    - Evaluate the model on test data
    - Save the trained model and evaluation metrics
    """

    def __init__(self, config_path: str = "config/config.yaml"):

        # Load configuration file (paths, filenames, settings)
        self.config = load_config(config_path)

        # Directory where processed training data exists (after preprocessing)
        self.processed_dir = os.path.abspath(self.config["data"]["processed_dir"])

        # Where to save trained model (models/)
        self.model_dir = os.path.abspath(self.config["model"]["model_dir"])
        self.model_file = self.config["model"]["model_file"]

        # Create model directory if missing
        os.makedirs(self.model_dir, exist_ok=True)

    # /// Step 1: Load processed dataset
    def load_processed_data(self) -> pd.DataFrame:
        """
        Loads the processed training dataset saved by data_preprocessor.py.

        Ensures:
        - File exists
        - 'SalePrice' column (target variable) exists
        """

        train_path = os.path.join(self.processed_dir, "train_processed.csv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Processed training file not found: {train_path}")

        df = pd.read_csv(train_path)

        if "SalePrice" not in df.columns:
            raise KeyError("Target column 'SalePrice' missing in processed dataset")

        print(f"Loaded processed data with shape: {df.shape}")
        return df

    # /// Step 2:  Split data into X (features) and y (target)
    def split_features_target(self, df: pd.DataFrame):
        """
        Splits the dataset into:
        - X: Feature matrix (all columns except SalePrice)
        - y: Target vector (SalePrice)
        Then performs train/test split.

        Purpose:
        - Train model on 80% of data
        - Test model performance on 20%
        """

        X = df.drop("SalePrice", axis=1)
        y = df["SalePrice"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # /// Step 3: Train RandomForest model
    def train_random_forest(self, X_train, y_train):
        """
        Initializes and trains a RandomForestRegressor.

        Why Random Forest?
        - Works well with large feature sets (193+)
        - No need for feature scaling
        - Handles nonlinear relationships
        - Robust to outliers
        """

        rf = RandomForestRegressor(
            n_estimators=300,         # number of trees
            max_depth=None,           # full tree depth
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1                 # use all CPU cores
        )

        rf.fit(X_train, y_train)
        print("Random Forest training completed.")
        return rf

    # ///Step 4: Evaluate model performance
    def evaluate_model(self, model, X_test, y_test) -> dict:
        """
        Evaluates trained model using:
        - RMSE (Root Mean Squared Error)
        - MAE  (Mean Absolute Error)
        - R²   (Explained variance)

        RMSE: How much the prediction deviates from the true value
        MAE : Average error in dollars
        R²  : How much variance in price the model explains
        """

        y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred)**0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE : {mae:.2f}")
        print(f"R²  : {r2:.4f}")

        return {"rmse": rmse, "mae": mae, "r2": r2}

    # /// Step 5: Save trained model
    def save_model(self, model):
        """
        Saves the trained model using joblib.
        This model is later used in prediction/app files.
        """

        model_path = os.path.join(self.model_dir, self.model_file)
        joblib.dump(model, model_path)

        print(f"Model saved to: {model_path}")

    # /// Step 6: Save metrics
    def save_metrics(self, metrics: dict):
        """
        Saves RMSE, MAE, R² results into a JSON file for tracking.
        """

        metrics_dir = os.path.abspath(self.config["evaluation"]["metrics_dir"])
        metrics_file = self.config["evaluation"]["metrics_file"]

        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, metrics_file)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics saved to: {metrics_path}")

    # /// Step 7: Full training pipeline
    def run(self):
        """
        Runs the full ML pipeline:
        1. Load processed dataset
        2. Split into train/test
        3. Train Random Forest
        4. Evaluate performance
        5. Save model and metrics
        """

        df = self.load_processed_data()
        X_train, X_test, y_train, y_test = self.split_features_target(df)

        model = self.train_random_forest(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test)

        self.save_model(model)
        self.save_metrics(metrics)

        return metrics


# Run directly
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()