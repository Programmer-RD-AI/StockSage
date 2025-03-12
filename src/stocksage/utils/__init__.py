"""
Utility functions for the StockSage application.

This module provides various utilities for data fetching, telemetry tracking,
and logging functionality needed throughout the application.
"""

from .api_fetch import (
    fetch_html,  # Fetches HTML content from a specified URL
    get_sp500_symbols,  # Returns a list of S&P 500 stock symbols
    get_nasdaq100_symbols,  # Returns a list of NASDAQ 100 stock symbols
    get_dow30_symbols,  # Returns a list of Dow Jones 30 stock symbols
    get_yfinance_data_sync,  # Synchronously retrieves data from Yahoo Finance
    get_yfinance_data,  # Asynchronously retrieves data from Yahoo Finance
    get_alpha_vantage_data,  # Retrieves financial data from Alpha Vantage API
)
from .telemetry_tracking import (
    initialize_event_loop,  # Initializes an event loop for asynchronous operations
    langsmith_task_callback,  # Callback function for LangSmith task tracking
    langsmith_step_callback,  # Callback function for LangSmith step tracking
    verify_langsmith_setup,  # Verifies that LangSmith is properly configured
)
from .logger import get_logger  # Returns a configured logger instance

__all__ = [
    "fetch_html",
    "get_sp500_symbols",
    "get_nasdaq100_symbols",
    "get_dow30_symbols",
    "get_yfinance_data_sync",
    "get_yfinance_data",
    "get_alpha_vantage_data",
    "initialize_event_loop",
    "langsmith_task_callback",
    "langsmith_step_callback",
    "verify_langsmith_setup",
    "get_logger",
]
