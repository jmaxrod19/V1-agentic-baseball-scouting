"""Shared pytest setup, auto-loaded before any test.

pytest imports this file first (it's named `conftest.py`, which pytest treats
specially). Its one job here: put the repo root on Python's import path so the
tests can do `from src import loaders, metrics, charts` no matter what directory
pytest is launched from.
"""

import sys
from pathlib import Path

# The repo root is the folder this file lives in.
_ROOT = Path(__file__).resolve().parent

# Prepend it to sys.path (the list of places Python looks for imports) if it
# isn't already there, so `import src` resolves to this project's src/ package.
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
