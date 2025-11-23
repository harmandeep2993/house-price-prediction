import os 
import yaml

# Define fucntion to load config.yaml
def load_config(config_path:str):
    
    """
    Load YAMl config file and return a Python dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)