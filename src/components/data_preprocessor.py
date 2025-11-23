import os
import pandas as pd
from src.utils.config_loader import load_config


class Datapreprocessor:
    """
    Reproduces the exact preprocessing logic used in EDA + FE notebook.
    Steps:
    - Drop unused columns
    - Fix missing values (LotFrontage, Garage, Basement, MasVnr, Electrical)
    - Apply ordinal mappings
    - One-hot encoding
    Produces ~193 final features after encoding.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)

        # Processed directory from config
        self.processed_dir = os.path.abspath(self.config["data"]["processed_dir"])
        os.makedirs(self.processed_dir, exist_ok=True)
   
    # /// Step 1: Drop unused columns
    @staticmethod
    def df_unused_columns(df: pd.DataFrame) -> pd.DataFrame:

        drop_cols = ["Id", "Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]

        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        return df

    # /// Step 2: Fix LotFrontage using neighborhood median
    @staticmethod
    def fix_lot_frontage(df: pd.DataFrame) -> pd.DataFrame:

        if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
            df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
                lambda x: x.fillna(x.median())
            )
        return df

    # /// Step 3: Fix GarageYrBlt + garage related missing values
    @staticmethod
    def fix_garage(df: pd.DataFrame) -> pd.DataFrame:

        garage_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]

        # First check if GarageYrBlt has missing values → fill with 0
        if df["GarageYrBlt"].isna().sum() > 0:
            df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

        # If GarageYrBlt == 0 → no garage → set garage categories to None
        no_garage_mask = df["GarageYrBlt"] == 0
        df.loc[no_garage_mask, garage_cols] = "None"

        # Fill any remaining missing categorical garage columns
        for col in garage_cols:
            if col in df.columns:
                df[col] = df[col].fillna("None")

        return df

    # /// Step 4: Fix MasVnrArea + MasVnrType
    @staticmethod
    def fix_mas_vnr(df: pd.DataFrame) -> pd.DataFrame:

        if "MasVnrArea" in df.columns:
            df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

        if "MasVnrType" in df.columns:
            df["MasVnrType"] = df["MasVnrType"].fillna("None")

        return df

    # /// Step 5: Fix Electrical (mode)
    @staticmethod
    def fix_electrical(df: pd.DataFrame) -> pd.DataFrame:
        if "Electrical" in df.columns:
            df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
        return df

    # /// Step 6: Full ordinal encoding (exact from notebook)
    @staticmethod
    def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:

        # Ordinal column list
        ordinal_cols = [
            'ExterQual','ExterCond',
            'BsmtQual','BsmtCond',
            'HeatingQC','KitchenQual',
            'GarageQual','GarageCond',
            'BsmtExposure','BsmtFinType1','BsmtFinType2',
            'LotShape','LandSlope',
            'Functional'
        ]

        # Keep only columns that exist
        ordinal_cols = [c for c in ordinal_cols if c in df.columns]

        # Mapping dictionaries
        quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0, "NA": 0}
        exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0, "NA": 0}
        bsmtfin_map = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0, "NA": 0}
        lotshape_map = {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1}
        landslope_map = {"Gtl": 3, "Mod": 2, "Sev": 1}
        functional_map = {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0}

        # Consolidated mapping
        ordinal_mappings = {
            'ExterQual': quality_map,
            'ExterCond': quality_map,
            'BsmtQual': quality_map,
            'BsmtCond': quality_map,
            'HeatingQC': quality_map,
            'KitchenQual': quality_map,
            'GarageQual': quality_map,
            'GarageCond': quality_map,
            'BsmtExposure': exposure_map,
            'BsmtFinType1': bsmtfin_map,
            'BsmtFinType2': bsmtfin_map,
            'LotShape': lotshape_map,
            'LandSlope': landslope_map,
            'Functional': functional_map
        }

        # Apply encoding
        for col in ordinal_cols:
            df[col] = df[col].map(ordinal_mappings[col])

        return df


    # /// Step 7: One-hot encoding of categorical features
    @staticmethod
    def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:

        if "MSSubClass" in df.columns:
            df["MSSubClass"] = df["MSSubClass"].astype("category")

        df = pd.get_dummies(df, drop_first=False)

        return df
    
    # /// Step 8: Convert all features to numeric
    @staticmethod
    def convert_df(df: pd.DataFrame) -> pd.DataFrame:
        
        for col in df.columns:
            # Convert bool features to 0 or 1
            if df[col].dtype == bool:

                df[col] = df[col].astype(int)
            
            elif df[col].dtype == "object":
                raise ValueError(f"Column '{col}' is still object type")

        return df

    # FULL PREPROCESSING PIPELINE
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:


        df = self.df_unused_columns(df)
        df = self.fix_lot_frontage(df)
        df = self.fix_garage(df)
        df = self.fix_mas_vnr(df)
        df = self.fix_electrical(df)

        df = self.encode_ordinal_features(df)
        df = self.encode_categoricals(df)
        df = self.convert_df(df)

        return df

    # Save processed CSV
    def save_processed(self, df: pd.DataFrame, filename="train_processed.csv"):

        output_path = os.path.join(self.processed_dir, filename)

        df.to_csv(output_path, index=False)
        print(f"Processed dataset saved to: {output_path}")

        return output_path


# Manual test
if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "train.csv")
    df_raw = pd.read_csv(raw_path)

    pre = Datapreprocessor()
    df_processed = pre.preprocess(df_raw)
    pre.save_processed(df_processed)