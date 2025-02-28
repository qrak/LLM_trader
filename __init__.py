from pathlib import Path
import os

# Create __init__.py files
project_root = Path(__file__).parent.parent
core_dir = project_root / "core"

# Create __init__.py in root and core directory
(project_root / "__init__.py").touch()
(core_dir / "__init__.py").touch()