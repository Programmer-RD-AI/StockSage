from stocksage.tools.stock_symbol_fetcher_tool import StockSymbolFetcherTool
from typing import Dict


def test_stock_symbol_fetcher_tool():
    stock_symbol_fetcher_tool = StockSymbolFetcherTool()
    data = stock_symbol_fetcher_tool.run()
    assert isinstance(data, Dict)
