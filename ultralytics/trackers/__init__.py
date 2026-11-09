# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bot_sort import BOTSORT
    from .byte_tracker import BYTETracker
    from .track import register_tracker

__all__ = "BOTSORT", "BYTETracker", "register_tracker"  # allow simpler import
_EXPORT_MODULES = {"BOTSORT": ".bot_sort", "BYTETracker": ".byte_tracker", "register_tracker": ".track"}


def __getattr__(name: str):
    # Importing a filter utility should not load the optional LAP-based trackers.
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value
