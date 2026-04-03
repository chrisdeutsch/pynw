"""
.. include:: ../README.md
   :start-line: 1
   :end-before: ## API
"""

from importlib.metadata import version

from pynw._native import needleman_wunsch
from pynw._ops import EditOp, alignment_indices

__docformat__ = "numpy"
__version__ = version("pynw")

__all__ = [
    "needleman_wunsch",
    "EditOp",
    "alignment_indices",
]
