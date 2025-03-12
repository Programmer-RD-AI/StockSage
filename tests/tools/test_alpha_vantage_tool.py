from stocksage.tools.alpha_vantage_tool import AlphaVantageTool
from typing import Dict


def test_alpha_vantage_tool():
    alpha_vantage_tool = AlphaVantageTool()
    data = alpha_vantage_tool.run("AAPL", "TIME_SERIES_DAILY")
    assert isinstance(data, Dict)
