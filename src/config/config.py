"""
Centralized configuration loader for the application.
Loads graph_config.yaml once and provides easy access to prompts with variable injection.
"""

import os
import re
from pathlib import Path
from typing import Any

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

    def _resolve_env_vars(self, value: Any) -> Any:
        """
        Recursively resolve environment variables in YAML values.
        Supports ${VAR} and ${VAR:-default} syntax.
        """
        if isinstance(value, str):
            # Match ${VAR} or ${VAR:-default}
            pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
            
            def replace_env(match):
                var_name = match.group(1)
                default = match.group(2) if match.group(2) else None
                env_value = os.getenv(var_name, default)
                return env_value if env_value is not None else match.group(0)
            
            return re.sub(pattern, replace_env, value)
        elif isinstance(value, dict):
            return {k: self._resolve_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_env_vars(item) for item in value]
        else:
            return value

    def _load_config(self):
        """Load the graph configuration from YAML file and validate with Pydantic."""
        # Get the root directory (ml-mcp)
        root_dir = Path(__file__).parent.parent.parent
        config_path = root_dir / self.CONFIG_FILENAME

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
                # Resolve environment variables
                yaml_data = self._resolve_env_vars(yaml_data)
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
