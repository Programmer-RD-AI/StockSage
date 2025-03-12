"""
StockSage: An intelligent stock analysis and selection system.

This package provides tools for automated stock analysis, market trend detection,
and investment recommendation using multi-agent LLM systems. StockSage combines
financial data analysis with AI-driven insights to help make informed investment decisions.

Classes:
    StockSage: Main application class that orchestrates the multi-agent system for stock analysis.

Functions:
    run: Execute the main StockSage application pipeline with specified parameters.
    test: Run evaluation and testing procedures on the StockSage system.
    train: Perform training or fine-tuning of the StockSage models and systems.
    replay: Replay historical analysis sessions for validation or demonstration.
"""

from .crew import StockSage  # Main application class for stock analysis
from .main import run, test, train, replay  # Core execution functions


__all__ = ["StockSage", "run", "test", "train", "replay"]
