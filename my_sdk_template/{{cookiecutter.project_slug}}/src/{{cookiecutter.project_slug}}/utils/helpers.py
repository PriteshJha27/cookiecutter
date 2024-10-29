import os
import yaml
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.yml
    """
    config_path = os.path.join(os.path.dirname(__file__), "../../config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)