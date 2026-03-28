"""
.. include:: ../README.md
   :start-line: 1
   :end-before: ## API
"""

from importlib.metadata import version

from pynw._native import needleman_wunsch

__all__ = ["needleman_wunsch"]
__docformat__ = "numpy"
__version__ = version("pynw")
