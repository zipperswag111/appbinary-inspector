"""Static binary inspection without executing the analyzed file."""

from .analyzer import inspect_file
from .extract import extract_binary_parts, extract_embedded
from .tokenizer import WordTokenizer

__all__ = ["inspect_file", "extract_embedded", "extract_binary_parts", "WordTokenizer"]
__version__ = "0.2.0"
