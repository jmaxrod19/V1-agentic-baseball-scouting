"""Baseball scouting toolkit package.

Re-exports the version so callers can read `src.__version__` (the Python
convention) while the value itself lives in config.py, the single source of
truth. Only runs when `src` is imported as a package; the modules' script-mode
fallback (`import config`) doesn't touch this file.
"""

from .config import APP_VERSION as __version__

__all__ = ["__version__"]
