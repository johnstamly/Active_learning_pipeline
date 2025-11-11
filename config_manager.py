import yaml
from typing import Dict, Any

class ConfigManager:
    """Utility class to load and access configuration from a YAML file."""
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads configuration from the specified YAML file path."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"Configuration loaded successfully from {config_path}")
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}")

    def get(self, key: str) -> Any:
        """Access a nested configuration value using a dot-separated key (e.g., 'PATHS.ARTIFACTS_DIR')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"Configuration key not found: {key}")
        return value

# Example: if you needed to use this utility in a standalone script:
# config = ConfigManager("config.yaml")
# artifact_dir = config.get("PATHS.ARTIFACTS_DIR")