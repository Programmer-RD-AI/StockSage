"""
This module contains the tools that are used to fetch data from various sources. The tools are used to fetch data from APIs, websites, and other sources. The tools are designed to be modular and reusable, so they can be easily integrated into other parts of the application.
To use a tool, you need to import it from the stocksage.tools module and create an instance of the tool. For example:
```
from stocksage.tools.alpha_vantage_tool import AlphaVantageTool

alpha_vantage_tool = AlphaVantageTool()
data = alpha_vantage_tool.run("AAPL", "TIME_SERIES_DAILY")
print(data)
```

The tools module contains the following tools:
- AlphaVantageTool: A tool to fetch data from the Alpha Vantage API.
- YFinanceTool: A tool to fetch data from the Yahoo Finance API.
- SentimentAnalysisTool: A tool to analyze sentiment of text data.
- StockSymbolFetcherTool: A tool to fetch stock symbols from various sources.
- get_firecrawl_crawl_website_tool: A tool to crawl websites using the FireCrawl tool.
- get_firecrawl_scrape_website_tool: A tool to scrape websites using the FireCrawl tool.
"""

from stocksage.tools.alpha_vantage_tool import AlphaVantageTool
from stocksage.tools.yahoo_finance_tool import YFinanceTool
from stocksage.tools.sentiment_analysis_tool import SentimentAnalysisTool
from stocksage.tools.stock_symbol_fetcher_tool import StockSymbolFetcherTool
from stocksage.tools.default_tools import (
    get_firecrawl_crawl_website_tool,
    get_firecrawl_scrape_website_tool,
)

__all__ = [
    "AlphaVantageTool",
    "YFinanceTool",
    "SentimentAnalysisTool",
    "StockSymbolFetcherTool",
    "get_firecrawl_crawl_website_tool",
    "get_firecrawl_scrape_website_tool",
]
