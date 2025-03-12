import asyncio
from crewai.tools import BaseTool
from typing import Type, Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from stocksage.utils import get_logger, get_yfinance_data

logger = get_logger()


class YFinanceInput(BaseModel):
    """
    Input schema for YFinanceTool.

    This model defines the required and optional parameters for making
    requests to the YFinanceTool, which fetches stock data from Yahoo Finance.

    Attributes:
        ticker (str): Stock ticker symbol (required)
        metrics (Optional[List[str]]): List of specific financial metrics to retrieve.
            Examples include PE (Price to Earnings), PB (Price to Book), ROE (Return on Equity).
            If None, returns all available metrics.
    """

    ticker: str = Field(..., description="Stock ticker symbol")
    metrics: Optional[List[str]] = Field(
        None, description="List of metrics to retrieve (e.g., PE, PB, ROE)"
    )


class YFinanceTool(BaseTool):
    """
    Tool for retrieving financial data from Yahoo Finance API.

    This tool fetches comprehensive financial information for stocks using the
    Yahoo Finance API. It provides access to both historical price data and
    fundamental financial metrics for analysis and evaluation.

    Attributes:
        name (str): Name of the tool
        description (str): Detailed description of the tool's capabilities
        args_schema (Type[BaseModel]): Schema for validating input arguments
    """

    name: str = "YFinance Stock Data Tool"
    description: str = (
        "Retrieves financial data for stocks using Yahoo Finance. "
        "Can provide metrics like PE ratio, PB ratio, market cap, revenue growth, "
        "profit margins, debt-to-equity, current ratio, 52-week price change, "
        "average volume, short interest, and volatility."
    )
    args_schema: Type[BaseModel] = YFinanceInput

    # @traceable
    def _run(self, ticker: str, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute the Yahoo Finance API request and process the response.

        Fetches both historical price data and company information for the specified
        stock ticker. Processes the information into a structured format containing
        key financial metrics and indicators.

        Args:
            ticker (str): Stock ticker symbol to fetch data for (e.g., 'AAPL', 'MSFT')
            metrics (Optional[List[str]], optional): Specific metrics to retrieve.
                If None, all available metrics will be returned. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary containing financial data including:
                - ticker: The requested ticker symbol
                - company_name: Full company name
                - current_price: Latest stock price
                - market_cap: Company market capitalization
                - pe_ratio: Price to earnings ratio
                - pb_ratio: Price to book ratio
                - revenue_growth: Revenue growth as percentage
                - profit_margin: Profit margin as percentage
                - debt_to_equity: Debt to equity ratio
                - current_ratio: Current ratio (liquidity measure)
                - 52w_change: 52-week price change percentage
                - average_volume: Average trading volume
                - short_interest: Short interest as percentage of float
                - volatility: Beta value indicating stock volatility

                In case of an error:
                - error: Error message

        Raises:
            No exceptions are raised directly as they're caught and returned as error responses.
        """
        try:
            with ThreadPoolExecutor() as executor:
                hist, info = asyncio.run(get_yfinance_data(ticker, executor))

            if hist.empty or not info:
                return {"error": f"Could not retrieve data for {ticker}"}

            data = {
                "ticker": ticker,
                "company_name": info.get("longName", ticker),
                "current_price": info.get("currentPrice"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "revenue_growth": f"{info.get('revenueGrowth', 0) * 100:.2f}%",
                "profit_margin": f"{info.get('profitMargins', 0) * 100:.2f}%",
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "52w_change": f"{info.get('52WeekChange', 0) * 100:.2f}%",
                "average_volume": info.get("averageVolume"),
                "short_interest": info.get("shortPercentOfFloat"),
                "volatility": info.get("beta"),
            }
            return data
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {ticker}: {e}")
            return {"error": f"Error fetching data for {ticker}: {e}"}
