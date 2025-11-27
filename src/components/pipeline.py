import os
import pandas as pd
from src.components.data_ingestor import DataIngestion
from src.components.data_preprocessor import Datapreprocessor
from src.components.model_trainer import ModelTrainer
from src.utils.config_loader import load_config

def main(config_path= "config/config.yaml"):
    config_path =load_config(config_path)

    # // Step 1: Load dataset
    print("Loading Data")
    ingestor= DataIngestion(config_path)
    df= ingestor.load_dataset()


    # // Step 2: Preprocess Data
    print("Preprocessing data...")
    preprocessor = Datapreprocessor(config_path)
    df_processed = preprocessor.preprocess(df_raw)
    preprocessor.save_processed(df_processed, filename="train_processed.csv")

    # Step 3: Train model
    print("Training model...")
    trainer = ModelTrainer(config_path)
    trainer.run()

    print("\n Pipeline completed successfully!")


if __name__ == "__main__":
    main()