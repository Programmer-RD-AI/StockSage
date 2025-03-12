from stocksage.utils.api_fetch import (
    fetch_html,
    get_sp500_symbols,
    get_nasdaq100_symbols,
    get_dow30_symbols,
    get_yfinance_data_sync,
    get_alpha_vantage_data,
)
import pandas as pd
import os
import pytest
from dotenv import load_dotenv
from aiohttp import ClientSession

load_dotenv()


@pytest.mark.asyncio
async def test_fetch_html():
    async with ClientSession() as session:
        result = await fetch_html("https://www.google.com", session)
        assert result is not None


@pytest.mark.asyncio
async def test_get_sp500_symbols():
    async with ClientSession() as session:
        symbols = await get_sp500_symbols(session)
        assert len(symbols) > 0


@pytest.mark.asyncio
async def test_get_nasdaq100_symbols():
    async with ClientSession() as session:
        symbols = await get_nasdaq100_symbols(session)
        assert len(symbols) > 0


@pytest.mark.asyncio
async def test_get_dow30_symbols():
    async with ClientSession() as session:
        symbols = await get_dow30_symbols(session)
        assert len(symbols) > 0


def test_get_yfinance_data_sync():
    data = get_yfinance_data_sync("AAPL")
    # Fix the type checking - check each component separately
    assert isinstance(data, tuple) and len(data) == 2
    assert isinstance(data[0], pd.DataFrame)
    assert isinstance(data[1], dict)
    assert data[1] != {}
    assert not data[0].empty


@pytest.mark.asyncio
async def test_get_alpha_vantage_data():
    async with ClientSession() as session:
        data = await get_alpha_vantage_data(
            "AAPL", os.getenv("ALPHA_VANTAGE_KEY"), session
        )
        assert isinstance(data, (pd.DataFrame, dict))
