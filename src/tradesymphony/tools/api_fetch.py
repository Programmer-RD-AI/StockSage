import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from tqdm import tqdm
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import Alpha Vantage if installed
try:
    from alpha_vantage.timeseries import TimeSeries

    alpha_vantage_available = True
except ImportError:
    logger.warning(
        "Alpha Vantage package not found. Use 'pip install alpha_vantage' to install it."
    )
    alpha_vantage_available = False


async def fetch_html(url, session):
    """Asynchronously fetch HTML content"""
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


async def get_sp500_symbols(session):
    """Get S&P 500 company symbols asynchronously"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        html = await fetch_html(url, session)
        if html:
            tables = pd.read_html(html)
            return tables[0]["Symbol"].tolist()
        return []
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {e}")
        return []


async def get_nasdaq100_symbols(session):
    """Get NASDAQ-100 company symbols asynchronously"""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        html = await fetch_html(url, session)
        if html:
            tables = pd.read_html(html)
            return tables[1]["Ticker"].tolist()
        return []
    except Exception as e:
        logger.error(f"Error fetching NASDAQ-100 symbols: {e}")
        return []


def get_yfinance_data_sync(symbol):
    """Synchronous function to get yfinance data (will be run in ThreadPoolExecutor)"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="5m")
        info = ticker.info
        return hist, info
    except Exception as e:
        logger.error(f"Error fetching yfinance data for {symbol}: {e}")
        return pd.DataFrame(), {}


async def get_yfinance_data(symbol, executor):
    """Get today's data from yfinance API asynchronously"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(get_yfinance_data_sync, symbol))


async def get_alphavantage_data(symbol, api_key, session):
    """Get today's data from Alpha Vantage API asynchronously"""
    if not alpha_vantage_available:
        return pd.DataFrame()

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&outputsize=compact&apikey={api_key}"
    try:
        async with session.get(url) as response:
            data = await response.json()

        if "Time Series (5min)" not in data:
            return pd.DataFrame()

        # Convert to pandas DataFrame
        time_series = data["Time Series (5min)"]
        df = pd.DataFrame.from_dict(time_series, orient="index")
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

        # Get only today's data
        today = datetime.now().strftime("%Y-%m-%d")
        df = df[df.index.astype(str).str.startswith(today)]
        return df
    except Exception as e:
        logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
        return pd.DataFrame()


async def get_finnhub_sentiment(symbol, api_key, session):
    """Get news sentiment from Finnhub API asynchronously"""
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={api_key}"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            return {}
    except Exception as e:
        logger.error(f"Error fetching Finnhub data for {symbol}: {e}")
        return {}


def calculate_analytics(df, info=None):
    """Calculate analytics for a stock based on its dataframe"""
    if df.empty:
        return {}

    analytics = {}

    # Basic price analytics
    if "Close" in df.columns and len(df) > 0:
        latest_price = df["Close"].iloc[-1]
        open_price = df["Open"].iloc[0]
        high = df["High"].max()
        low = df["Low"].min()
        price_change = latest_price - open_price
        price_change_pct = (price_change / open_price) * 100

        analytics.update(
            {
                "latest_price": latest_price,
                "open_price": open_price,
                "high": high,
                "low": low,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
            }
        )

    # Volume analytics
    if "Volume" in df.columns and len(df) > 0:
        total_volume = df["Volume"].sum()
        avg_volume = df["Volume"].mean()

        analytics.update(
            {
                "total_volume": total_volume,
                "avg_volume": avg_volume,
            }
        )

    # Technical indicators
    if len(df) > 20 and "Close" in df.columns:
        # Moving Averages
        df["SMA_5"] = df["Close"].rolling(window=5).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()

        # MACD (Moving Average Convergence Divergence)
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

        # RSI (Relative Strength Index)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        analytics.update(
            {
                "SMA_5": df["SMA_5"].iloc[-1],
                "SMA_20": df["SMA_20"].iloc[-1] if len(df) >= 20 else None,
                "MACD": df["MACD"].iloc[-1],
                "MACD_signal": df["MACD_signal"].iloc[-1],
                "RSI": df["RSI"].iloc[-1],
            }
        )

    # Add fundamental data if available
    if info:
        fundamental_metrics = [
            "marketCap",
            "trailingPE",
            "forwardPE",
            "dividendYield",
            "beta",
            "52WeekChange",
            "shortPercentOfFloat",
        ]
        for metric in fundamental_metrics:
            if metric in info:
                analytics[metric] = info[metric]

    return analytics


async def process_stock(
    symbol,
    av_key,
    finnhub_key,
    executor,
    session,
    semaphore_av,
    semaphore_finnhub,
    save_raw_data,
):
    """Process a single stock asynchronously"""
    try:
        # Get data from yfinance (runs in thread pool since yfinance is synchronous)
        yf_data, yf_info = await get_yfinance_data(symbol, executor)

        # Get data from Alpha Vantage if available (with rate limiting)
        av_data = pd.DataFrame()
        if av_key:
            async with semaphore_av:
                av_data = await get_alphavantage_data(symbol, av_key, session)

        # Get sentiment data from Finnhub if available (with rate limiting)
        sentiment_data = {}
        if finnhub_key:
            async with semaphore_finnhub:
                sentiment_data = await get_finnhub_sentiment(
                    symbol, finnhub_key, session
                )

        # Calculate analytics (synchronous operation but fast)
        yf_analytics = calculate_analytics(yf_data, yf_info)
        av_analytics = calculate_analytics(av_data)

        # Combine all analytics
        combined_analytics = {
            "Symbol": symbol,
            "Date": datetime.now().strftime("%Y-%m-%d"),
        }

        # Add yfinance analytics
        combined_analytics.update({f"yf_{k}": v for k, v in yf_analytics.items()})

        # Add Alpha Vantage analytics
        combined_analytics.update({f"av_{k}": v for k, v in av_analytics.items()})

        # Add sentiment data
        if sentiment_data and "sentiment" in sentiment_data:
            combined_analytics.update(
                {
                    "sentiment_score": sentiment_data.get("sentiment", {}).get("score"),
                    "sentiment_bullish": sentiment_data.get("sentiment", {}).get(
                        "bullishPercent"
                    ),
                    "sentiment_bearish": sentiment_data.get("sentiment", {}).get(
                        "bearishPercent"
                    ),
                }
            )

        # Return results and raw data
        raw_data_result = (
            {
                "yfinance": yf_data,
                "alpha_vantage": av_data,
                "sentiment": sentiment_data,
            }
            if save_raw_data
            else {}
        )

        return combined_analytics, raw_data_result, None

    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None, None, str(e)


async def analyze_us_stocks_async(max_stocks=None, save_raw_data=False):
    """Asynchronous main function to analyze US stocks"""
    # Get API keys from environment variables
    av_key = os.environ.get("ALPHA_VANTAGE_KEY", "2AMJW2DXVLR8KEEO")
    finnhub_key = os.environ.get(
        "FINNHUB_KEY", "cv4in49r01qn2gaadb80cv4in49r01qn2gaadb8g"
    )

    # Display API key status
    logger.info("\n--- API Status ---")
    logger.info(f"Alpha Vantage API key: {'Available' if av_key else 'Missing'}")
    logger.info(f"Finnhub API key: {'Available' if finnhub_key else 'Missing'}")

    # Create semaphores for API rate limiting
    # Alpha Vantage allows 5 requests per minute (1 every 12 seconds to be safe)
    semaphore_av = asyncio.Semaphore(5)
    # Finnhub allows 60 requests per minute (1 every second to be safe)
    semaphore_finnhub = asyncio.Semaphore(30)

    # Create thread pool for yfinance (which is not async-compatible)
    executor = ThreadPoolExecutor(max_workers=10)

    # Create aiohttp session for all HTTP requests
    async with aiohttp.ClientSession() as session:
        # Get list of stock symbols asynchronously
        logger.info("\nFetching stock symbols...")
        symbols = await get_sp500_symbols(session)

        if not symbols:
            logger.info("Trying NASDAQ-100 symbols...")
            symbols = await get_nasdaq100_symbols(session)

        if not symbols:
            logger.info("Using default list of major stocks")
            symbols = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
                "TSLA",
                "NVDA",
                "JPM",
                "V",
                "WMT",
            ]

        # Limit number of stocks if specified
        if max_stocks and max_stocks < len(symbols):
            symbols = symbols[:max_stocks]

        logger.info(f"Analyzing {len(symbols)} stocks...")

        # Create tasks for all stocks
        tasks = [
            process_stock(
                symbol,
                av_key,
                finnhub_key,
                executor,
                session,
                semaphore_av,
                semaphore_finnhub,
                save_raw_data,
            )
            for symbol in symbols
        ]

        # Process all stocks in parallel with progress tracking
        all_results = []
        raw_data = {}
        progress_bar = tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Processing stocks"
        )

        for future in progress_bar:
            analytics, stock_raw_data, error = await future
            if analytics:
                all_results.append(analytics)
                if save_raw_data and stock_raw_data:
                    raw_data[analytics["Symbol"]] = stock_raw_data

    # Create DataFrame from results
    all_results_df = pd.DataFrame(all_results)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stock_analytics_{timestamp}.csv"
    all_results_df.to_csv(filename, index=False)

    logger.info(f"\nAnalysis complete! Results saved to {filename}")

    # Print summary of top gainers and losers
    if len(all_results_df) > 0:
        logger.info("\n--- TOP GAINERS ---")
        gainers = all_results_df.sort_values(
            "yf_price_change_pct", ascending=False
        ).head(5)
        print(gainers[["Symbol", "yf_latest_price", "yf_price_change_pct"]])

        logger.info("\n--- TOP LOSERS ---")
        losers = all_results_df.sort_values("yf_price_change_pct").head(5)
        print(losers[["Symbol", "yf_latest_price", "yf_price_change_pct"]])

    return all_results_df, raw_data


def analyze_us_stocks(max_stocks=None, save_raw_data=False):
    """Wrapper function to run async code"""
    return asyncio.run(analyze_us_stocks_async(max_stocks, save_raw_data))


if __name__ == "__main__":
    print("=" * 50)
    print("ASYNC US STOCK MARKET ANALYTICS")
    print("=" * 50)
    analyze_us_stocks()
