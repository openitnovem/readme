"""
**icecream** explains how a machine learning model works using
Partial Dependency Plots and various Individual Conditional Expectation plots
"""

from .icecream import IceCream, IceCream2D
from .config import options

__all__ = ["IceCream", "IceCream2D", "options"]
