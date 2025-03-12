import yfinance as yf
import pandas as pd
import asyncio
from functools import partial
from .logger import get_logger
from typing import Optional, List, Tuple
import aiohttp
import concurrent


logger = get_logger()


async def fetch_html(url: str, session: aiohttp.ClientSession) -> str | None:
    """
    Asynchronously fetch HTML content from a URL.

    Makes an HTTP GET request to the specified URL and returns the HTML content
    as text. Designed to be used with an existing aiohttp ClientSession.

    Args:
        url (str): The URL to fetch HTML content from
        session (aiohttp.ClientSession): Active aiohttp session for making HTTP requests

    Returns:
        str or None: HTML content as a string if successful, None if an error occurs

    Raises:
        No exceptions are raised directly; errors are caught, logged, and None is returned
    """
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


async def get_sp500_symbols(session: aiohttp.ClientSession) -> List[str]:
    """
    Get S&P 500 company ticker symbols asynchronously.

    Fetches the current list of S&P 500 companies from Wikipedia and extracts
    their ticker symbols from the page's HTML table.

    Args:
        session (aiohttp.ClientSession): Active aiohttp session for making HTTP requests

    Returns:
        list[str]: List of stock ticker symbols for S&P 500 companies,
                  or an empty list if retrieval fails

    Note:
        Uses pandas to parse HTML tables from the Wikipedia page
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        html = await fetch_html(url, session)
        if html:
            from io import StringIO

            tables = pd.read_html(StringIO(html))
            return tables[0]["Symbol"].tolist()
        return []
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        return []


async def get_nasdaq100_symbols(session: aiohttp.ClientSession) -> List[str]:
    """
    Get NASDAQ-100 company ticker symbols asynchronously.

    Fetches the current list of NASDAQ-100 companies from Wikipedia and extracts
    their ticker symbols from the page's HTML tables. The function searches through
    all tables to find the one containing ticker symbols.

    Args:
        session (aiohttp.ClientSession): Active aiohttp session for making HTTP requests

    Returns:
        list[str]: List of stock ticker symbols for NASDAQ-100 companies,
                  or an empty list if retrieval fails

    Note:
        Uses pandas to parse HTML tables and searches for columns with
        names like "ticker", "symbol", "trade", or "code"
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        html = await fetch_html(url, session)
        if html:
            # Use StringIO to fix the pandas warning
            from io import StringIO

            tables = pd.read_html(StringIO(html))
            # Look through all tables for one with ticker symbols
            for i, table in enumerate(tables):
                columns = table.columns.tolist()
                # Check for various possible column names that might contain ticker symbols
                ticker_cols = [
                    col
                    for col in columns
                    if any(
                        name in str(col).lower()
                        for name in ["ticker", "symbol", "trade", "code"]
                    )
                ]
                if ticker_cols:
                    return table[ticker_cols[0]].tolist()
            return []
        return []
    except Exception as e:
        print(f"Error fetching NASDAQ-100 symbols: {e}")
        return []


async def get_dow30_symbols(session: aiohttp.ClientSession) -> List[str]:
    """
    Get Dow Jones Industrial Average (DJIA) ticker symbols asynchronously.

    Fetches the current list of 30 companies in the Dow Jones Industrial Average
    from Wikipedia and extracts their ticker symbols from the page's HTML tables.

    Args:
        session (aiohttp.ClientSession): Active aiohttp session for making HTTP requests

    Returns:
        list[str]: List of stock ticker symbols for the 30 DJIA companies,
                  or an empty list if retrieval fails

    Note:
        Searches through tables to find the one with a "Symbol" column
    """
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    try:
        html = await fetch_html(url, session)
        if html:
            from io import StringIO

            tables = pd.read_html(StringIO(html))
            # Find the table with company symbols
            for table in tables:
                if "Symbol" in table.columns:
                    return table["Symbol"].tolist()
            return []
        return []
    except Exception as e:
        print(f"Error fetching Dow 30 symbols: {e}")
        return []


def get_yfinance_data_sync(symbol: str) -> Tuple[pd.DataFrame, dict]:
    """
    Synchronous function to get financial data for a stock using yfinance.

    Fetches both historical price data and company information for the given
    ticker symbol. Designed to be used with a ThreadPoolExecutor for non-blocking
    operation.

    Args:
        symbol (str): Stock ticker symbol to fetch data for (e.g., 'AAPL', 'MSFT')

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: Historical price data for the stock
            - dict: Company information and financial metrics

        If an error occurs, returns an empty DataFrame and empty dict.

    Note:
        Errors are caught, logged, and empty values are returned
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="5m")
        info = ticker.info
        return hist, info
    except Exception as e:
        logger.error(f"Error fetching yfinance data for {symbol}: {e}")
        return pd.DataFrame(), {}


async def get_yfinance_data(
    symbol: str, executor: concurrent.futures.ThreadPoolExecutor
) -> Tuple[pd.DataFrame, dict]:
    """
    Get stock data from Yahoo Finance asynchronously.

    Wraps the synchronous yfinance API call in an asynchronous function by
    running it in a ThreadPoolExecutor to avoid blocking the event loop.

    Args:
        symbol (str): Stock ticker symbol to fetch data for (e.g., 'AAPL', 'MSFT')
        executor (concurrent.futures.ThreadPoolExecutor): Executor to run the
            synchronous yfinance call in

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: Historical price data for the stock
            - dict: Company information and financial metrics
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(get_yfinance_data_sync, symbol))


async def get_alpha_vantage_data(
    symbol: str,
    api_key: str,
    session: aiohttp.ClientSession,
    function: Optional[str] = "TIME_SERIES_INTRADAY",
    interval: Optional[str] = "5min",
) -> pd.DataFrame | dict:
    """
    Get financial data from Alpha Vantage API asynchronously.

    Fetches various types of financial data based on the specified function type.
    Supports time series data (intraday, daily), company overviews, and other
    Alpha Vantage API functions.

    Args:
        symbol (str): Stock ticker symbol to fetch data for (e.g., 'AAPL', 'MSFT')
        api_key (str): Alpha Vantage API key
        session (aiohttp.ClientSession): Active aiohttp session for making HTTP requests
        function (str, optional): Alpha Vantage API function to call. Defaults to "TIME_SERIES_INTRADAY".
            Common values: "TIME_SERIES_INTRADAY", "TIME_SERIES_DAILY", "OVERVIEW"
        interval (str, optional): Data interval for intraday data. Defaults to "5min".
            Valid values: "1min", "5min", "15min", "30min", "60min"

    Returns:
        pandas.DataFrame or dict:
            For time series data: A DataFrame with columns for OHLCV data
            For non-time series data (e.g., OVERVIEW): Original JSON response as dict
            If an error occurs: Empty DataFrame

    Note:
        For time series data, the function converts the JSON response to a DataFrame
        and attempts to rename columns to standard OHLCV format
    """

    base_url = "https://www.alphavantage.co/query"
    params = {"function": function, "symbol": symbol, "apikey": api_key}

    # Add interval parameter for intraday data
    if function == "TIME_SERIES_INTRADAY":
        params["interval"] = interval
        params["outputsize"] = "compact"

    # Build URL with parameters
    url_params = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{base_url}?{url_params}"

    try:
        async with session.get(url) as response:
            data = await response.json()

        # Handle error responses
        if "Error Message" in data:
            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            return pd.DataFrame()

        # Handle different function responses
        if function == "TIME_SERIES_INTRADAY":
            time_series_key = f"Time Series ({interval})"
        elif function == "TIME_SERIES_DAILY":
            time_series_key = "Time Series (Daily)"
        elif function == "OVERVIEW":
            # For company overview, just return the data as is
            return data
        else:
            # For other functions, return whatever data we get
            return data

        # If we have time series data, convert to DataFrame
        if time_series_key in data:
            # Convert to pandas DataFrame
            time_series = data[time_series_key]
            df = pd.DataFrame.from_dict(time_series, orient="index")

            # Rename columns based on common patterns
            if len(df.columns) >= 5:  # OHLCV data
                df.columns = ["Open", "High", "Low", "Close", "Volume"]
                df = df.astype(
                    {
                        "Open": float,
                        "High": float,
                        "Low": float,
                        "Close": float,
                        "Volume": float,
                    }
                )

            df.index = pd.DatetimeIndex(df.index)
            return df

        return data
    except Exception as e:
        logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
        return pd.DataFrame()
