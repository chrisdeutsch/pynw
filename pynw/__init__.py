"""
.. include:: ../README.md
   :start-line: 1
   :end-before: ## API
"""

from importlib.metadata import version

from pynw._native import needleman_wunsch, needleman_wunsch_merge_split
from pynw._ops import Op, indices_from_ops, iter_alignment

__docformat__ = "numpy"
__version__ = version("pynw")

__all__ = [
    "needleman_wunsch",
    "needleman_wunsch_merge_split",
    "Op",
    "indices_from_ops",
    "iter_alignment",
]
