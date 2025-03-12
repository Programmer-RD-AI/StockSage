from stocksage.tools.yahoo_finance_tool import YFinanceTool
from typing import Dict


def test_yahoo_finance_tool():
    yahoo_finance_tool = YFinanceTool()
    data = yahoo_finance_tool.run("AAPL")
    assert isinstance(data, Dict)
