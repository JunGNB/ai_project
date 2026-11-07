"""Result analysis tools for run bundles (compare, aggregate, export)."""

from .cli import main
from .compare import compare_summaries, load_summary, print_comparison

__all__ = ["compare_summaries", "load_summary", "print_comparison", "main"]
