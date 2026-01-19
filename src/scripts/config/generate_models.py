#!/usr/bin/env python3
"""
Generate Pydantic models from graph_config.yaml
This script creates src/config_models.py with type-safe models
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Get paths
    root_dir = Path(__file__).parent.parent.parent.parent
    yaml_path = root_dir / "graph_config.yaml"
    output_path = root_dir / "src" / "config" / "config_models.py"

    if not yaml_path.exists():
        print(f"❌ Config file not found: {yaml_path}")
        sys.exit(1)

    print(f"📄 Generating models from {yaml_path.name}...")

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
        print(f"✅ Models generated successfully at {output_path}")
        print("📝 You can now import from: from src.config_models import GraphConfig")
        return 0
    except subprocess.CalledProcessError as e:
        print("❌ Error generating models:")
        print(e.stderr)
        return 1
    except FileNotFoundError:
        print("❌ datamodel-codegen not found. Install it with:")
        print("   uv add datamodel-code-generator")
        return 1


if __name__ == "__main__":
    sys.exit(main())
