#!/usr/bin/env python3
"""
Generate Pydantic models from graph_config.yaml
This script creates src/config_models.py with type-safe models
"""

import logging
import subprocess
import sys
from pathlib import Path

from src.config.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main():
    configure_logging()

    # Get paths
    root_dir = Path(__file__).parent.parent.parent.parent
    yaml_path = root_dir / "graph_config.yaml"
    output_path = root_dir / "src" / "config" / "config_models.py"

    if not yaml_path.exists():
        logger.error("Config file not found: %s", yaml_path)
        sys.exit(1)

    logger.info("Generating models from %s", yaml_path.name)

    # Run datamodel-code-generator
    cmd = [
        "datamodel-codegen",
        "--input",
        str(yaml_path),
        "--output",
        str(output_path),
        "--input-file-type",
        "yaml",
        "--output-model-type",
        "pydantic_v2.BaseModel",
        "--field-constraints",
        "--use-default",
        "--use-standard-collections",
        "--use-annotated",
        "--collapse-root-models",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Models generated successfully at %s", output_path)
        logger.info("You can now import from: from src.config_models import GraphConfig")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error("Error generating models: %s", e.stderr)
        return 1
    except FileNotFoundError:
        logger.error(
            "datamodel-codegen not found. Install it with: uv add datamodel-code-generator"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
