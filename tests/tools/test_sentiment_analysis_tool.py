from stocksage.tools.sentiment_analysis_tool import SentimentAnalysisTool
from typing import Dict


def test_sentiment_analysis_tool():
    sentiment_analysis_tool = SentimentAnalysisTool()
    data = sentiment_analysis_tool.run("AAPL")
    assert isinstance(data, Dict)
