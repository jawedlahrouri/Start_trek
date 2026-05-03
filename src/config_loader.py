import yaml
import os

def load_config(config_path="configs/default_config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found : {config_path}")
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file : {exc}")
            return None
