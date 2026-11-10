"""Static binary inspection without executing the analyzed file."""

from .analyzer import inspect_file
from .extract import extract_embedded

__all__ = ["inspect_file", "extract_embedded"]
__version__ = "0.1.0"