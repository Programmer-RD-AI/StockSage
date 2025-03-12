import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from stocksage.utils import (
    get_sp500_symbols,
    get_nasdaq100_symbols,
    get_dow30_symbols,
)


class StockSymbolRequest(BaseModel):
    """
    Request model for stock symbol fetching.

    This Pydantic model defines the schema for requests to the StockSymbolFetcherTool,
    validating input parameters and providing default values.

    Attributes:
        source (str): Source to fetch stock symbols from. Options include 'sp500',
            'nasdaq100', 'dow30', 'all', or 'custom'. Defaults to 'sp500'.
        custom_symbols (Optional[List[str]]): List of custom stock ticker symbols to use
            when source is set to 'custom'. Defaults to an empty list.
        limit (Optional[int]): Maximum number of symbols to return in the response.
            Defaults to 5.
    """

    source: str = Field(
        default="sp500",
        description="Source of stock symbols: 'sp500', 'nasdaq100', 'dow30', 'all', or 'custom'",
    )
    custom_symbols: Optional[List[str]] = Field(
        default=[],
        description="Custom list of stock symbols to use when source is 'custom'",
    )
    limit: Optional[int] = Field(
        default=5, description="Maximum number of symbols to return"
    )

    class Config:
        extra = "ignore"


class StockSymbolFetcherTool(BaseTool):
    """
    Tool for fetching stock symbols from various sources.

    This tool provides functionality to retrieve stock ticker symbols from different
    sources including S&P 500, NASDAQ-100, Dow Jones Industrial Average, or a custom list.
    It can fetch symbols asynchronously using web scraping from Wikipedia or return
    predefined lists.

    Attributes:
        name (str): Display name of the tool
        description (str): Brief description of the tool's functionality
    """

    name: str = "StockSymbolFetcherTool"
    description: str = (
        "Fetches stock symbols from various sources like S&P 500, NASDAQ-100, etc."
    )

    # @traceable
    async def get_default_symbols(self) -> List[str]:
        """
        Return a default list of major stock symbols.

        Provides a fallback list of 10 major company ticker symbols when other sources
        are not available or not requested.

        Returns:
            List[str]: List of 10 major stock ticker symbols including AAPL, MSFT, etc.
        """
        return [
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

    # @traceable
    async def _arun(
        self,
        source: Optional[str] = "sp500",
        custom_symbols: Optional[List[str]] = [],
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously fetch stock symbols from the specified source.

        This is the core async implementation of the tool that fetches stock symbols
        from various sources and processes them according to the provided parameters.

        Args:
            source (str): Source of stock symbols ('sp500', 'nasdaq100', 'dow30', 'all', 'custom').
                Supports various formats like 'S&P 500', 'NASDAQ-100', etc.
            custom_symbols (List[str], optional): Custom list of stock symbols when source is 'custom'.
                Defaults to [].
            limit (int, optional): Maximum number of symbols to return. Defaults to None (no limit).

        Returns:
            Dict[str, Any]: Dictionary containing:
                - symbols: List of stock ticker symbols
                - source: The source used (normalized)
                - count: Number of symbols returned
                - limited: Boolean flag if results were limited (when applicable)
                - details: Additional statistics (when source is 'all')
        """
        import aiohttp

        # Normalize source input to handle user-friendly names
        if source.lower() in ["s&p 500", "s&p500", "sp 500"]:
            source = "sp500"
        elif source.lower() in ["nasdaq 100", "nasdaq-100"]:
            source = "nasdaq100"
        elif source.lower() in ["dow 30", "dow jones", "dow"]:
            source = "dow30"
        results = {"symbols": [], "source": source}

        async with aiohttp.ClientSession() as session:
            if source == "custom" and custom_symbols:
                results["symbols"] = custom_symbols

            elif source == "all":
                sp500 = await get_sp500_symbols(session)
                nasdaq100 = await get_nasdaq100_symbols(session)
                dow30 = await get_dow30_symbols(session)

                # Combine all symbols (removing duplicates)
                all_symbols = list(set(sp500 + nasdaq100 + dow30))
                results["symbols"] = all_symbols
                results["details"] = {
                    "sp500_count": len(sp500),
                    "nasdaq100_count": len(nasdaq100),
                    "dow30_count": len(dow30),
                    "total_unique": len(all_symbols),
                }

            elif source == "sp500":
                results["symbols"] = await get_sp500_symbols(session)

            elif source == "nasdaq100":
                results["symbols"] = await get_nasdaq100_symbols(session)

            elif source == "dow30":
                results["symbols"] = await get_dow30_symbols(session)

            else:
                # Default to a small list of major stocks
                results["symbols"] = await self.get_default_symbols()
                results["source"] = "default"

        # Apply limit if specified
        if limit and len(results["symbols"]) > limit:
            results["symbols"] = results["symbols"][:limit]
            results["limited"] = True

        results["count"] = len(results["symbols"])
        return results

    # @traceable(run_type="tool")
    def _run(
        self,
        source: Optional[str] = "sp500",
        custom_symbols: Optional[List[str]] = [],
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for _arun method.

        Provides a synchronous interface to the asynchronous implementation by running
        the async method in an event loop.

        Args:
            source (str, optional): Source of stock symbols ('sp500', 'nasdaq100', 'dow30', 'all', 'custom').
                Defaults to "sp500".
            custom_symbols (List[str], optional): Custom list of stock symbols when source is 'custom'.
                Defaults to [].
            limit (int, optional): Maximum number of symbols to return. Defaults to None (no limit).

        Returns:
            Dict[str, Any]: Dictionary containing stock symbols and metadata as returned by _arun.

        Note:
            This method handles None for custom_symbols by converting it to an empty list.
        """
        # Ensure custom_symbols is an empty list if it's None
        custom_symbols = custom_symbols if custom_symbols is not None else []
        return asyncio.run(self._arun(source, custom_symbols, limit))


if __name__ == "__main__":
    StockSymbolFetcherTool().run(source="S&P 500", limit=15)
