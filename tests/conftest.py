import sys
from pathlib import Path

# Ensure the project root is on sys.path so `import cupel` works
# when running pytest from the repo root without pip install -e .
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
