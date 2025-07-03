"""
Welcome to ``python-Wappalyzer`` API documentation!

:see: `Wappalyzer` and `WebPage`.
"""

from .Wappalyzer import Wappalyzer, OptimizedWappalyzer, analyze
from .webpage import WebPage

__all__ = ["Wappalyzer", "OptimizedWappalyzer", "WebPage", "analyze"]
