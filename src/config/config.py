"""
Centralized configuration loader for the application.
Loads graph_config.yaml once and provides easy access to prompts with variable injection.
"""

from pathlib import Path

import yaml

# Import auto-generated models from config_models.py
# To regenerate: just generate-models
from src.config.config_models import Model as GraphConfig


class _ConfigLoader:
    """Internal singleton configuration loader."""

    CONFIG_FILENAME = "graph_config.yaml"

    _instance: "_ConfigLoader | None" = None
    _config: GraphConfig | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Load the graph configuration from YAML file and validate with Pydantic."""
        # Get the root directory (ml-mcp)
        root_dir = Path(__file__).parent.parent.parent
        config_path = root_dir / self.CONFIG_FILENAME

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
                # Parse and validate with Pydantic
                self._config = GraphConfig(**yaml_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")

    def get(self) -> GraphConfig:
        """Get the loaded configuration."""
        if self._config is None:
            self._load_config()
        return self._config  # type: ignore

    def reload(self):
        """Reload the configuration from disk."""
        self._load_config()


# Convenience function for easy access with full autocomplete
def get_config() -> GraphConfig:
    """Get the singleton configuration instance with full IDE autocomplete support."""
    return _ConfigLoader().get()
