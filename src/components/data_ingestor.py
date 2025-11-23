import os
import pandas as pd
from src.utils.config_loader import load_config


class DataIngestion:
    """
    Handles reading raw data and saving processed data.
    Reads all file paths from config/config.yaml.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        # Load config file
        self.config = load_config(config_path)

        # Validate required config keys
        if "data" not in self.config or "raw_path" not in self.config["data"]:
            raise KeyError("Missing 'data.raw_path' in config.yaml")

        # Convert to absolute paths to avoid relative path issues
        self.raw_data_path = os.path.abspath(self.config["data"]["raw_path"])
        self.processed_dir = os.path.abspath(self.config["data"]["processed_path"])

    def __repr__(self) -> str:
        # Helpful when debugging
        return f"DataIngestion(raw_path='{self.raw_data_path}')"

    def load_dataset(self) -> pd.DataFrame:
        """
        Loads the raw dataset from CSV.
        """
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"File not found: {self.raw_data_path}")

        df = pd.read_csv(self.raw_data_path)
        return df

    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Saves cleaned/processed data to the processed directory.
        Creates folder if missing.
        """
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

    @staticmethod
    def df_overview(df: pd.DataFrame, show: bool = True) -> dict:
        """
        Provides basic dataset diagnostics:
        - shape
        - dtypes
        - top missing-value columns
        """
        summary = {
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "missing_top20": (df.isna().sum().sort_values(ascending=False).head(20).to_dict())
            }

        if show:
            print("\n=== Shape ===", summary["shape"])
            print("\n=== Missing (Top 20 Columns) ===")
            print(df.isna().sum().sort_values(ascending=False).head(20))

        return summary

if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.load_dataset()
    ingestion.df_overview(df)