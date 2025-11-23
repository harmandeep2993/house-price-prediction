import os
import pandas as pd
import joblib
import json
from src.utils.config_loader import load_config


class ModelPredictor:

    def __init__(self, config_path="config/config.yaml"):
        
        # Load config
        self.config = load_config(config_path)

        # Read model_dir + model_file from config
        model_dir = self.config["model"]["model_dir"]
        model_file = self.config["model"]["model_file"]

        self.model_path = os.path.join(model_dir, model_file)

        # Check if model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load model
        self.model = joblib.load(self.model_path)
        print(f"Loaded model: {self.model_path}")

        # Load feature columns file
        self.columns_path = os.path.join(model_dir, "columns.json")

        if not os.path.exists(self.columns_path):
            raise FileNotFoundError(
                f"Feature column file missing: {self.columns_path}"
            )

        with open(self.columns_path, "r") as f:
            self.feature_columns = json.load(f)

        print(f"Loaded {len(self.feature_columns)} training columns.")


    def preprocess_input(self, input_dict):

        df = pd.DataFrame([input_dict])

        # Add missing columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Ensure column order
        df = df[self.feature_columns]

        return df


    def predict(self, input_dict):

        X = self.preprocess_input(input_dict)
        pred = self.model.predict(X)[0]
        return float(pred)


if __name__ == "__main__":
    predictor = ModelPredictor()

    # Example placeholder input (must match one-hot encoded structure)
    example_input = {
        "OverallQual": 7,
        "GrLivArea": 1800,
        "GarageCars": 2,
        "GarageArea": 500,
        "YearBuilt": 2005,
        # Any missing fields will be auto-filled with 0
    }

    price = predictor.predict(example_input)
    print(f"Predicted SalePrice: {round(price, 2)}")