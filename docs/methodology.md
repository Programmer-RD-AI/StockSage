# Methodology

This document details StockSage's approach to selecting the top 5 US stocks using a hierarchical multi-agent system.

## Stock Universe Definition

StockSage begins with a defined universe of stocks:

- **Markets**: NYSE and NASDAQ
- **Market Capitalization**: Large-cap ($10B+) and mid-cap ($2B-$10B)
- **Liquidity**: Minimum average daily trading volume of 100,000 shares
- **History**: Minimum of 5 years of trading history
- **Sectors**: All major S&P sectors included

## Data Collection Process

### Market Data

- Daily OHLCV (Open, High, Low, Close, Volume) data
- Intraday price movements (15-minute intervals)
- Historical price data (up to 10 years)

### Fundamental Data

- Quarterly financial statements
- Key financial ratios and metrics
- Earnings reports and surprises
- Analyst estimates and revisions

### Alternative Data

- News sentiment from financial publications
- Social media sentiment analysis
- Insider trading activity
- ESG (Environmental, Social, Governance) metrics
- Patent filings and R&D expenditures

## Multi-Factor Analysis Framework

StockSage employs a comprehensive multi-factor approach:

### Value Factors

- Price-to-Earnings (P/E) ratio
- Price-to-Book (P/B) ratio
- Price-to-Sales (P/S) ratio
- EV/EBITDA
- Free Cash Flow Yield

### Growth Factors

- Revenue growth (1-year, 3-year, 5-year)
- Earnings growth (1-year, 3-year, 5-year)
- Cash flow growth
- Profit margin expansion
- R&D growth rate

### Quality Factors

- Return on Equity (ROE)
- Return on Assets (ROA)
- Debt-to-Equity ratio
- Interest coverage ratio
- Earnings stability

### Momentum Factors

- Price momentum (3-month, 6-month, 12-month)
- Earnings momentum
- Analyst revision momentum
- Relative strength vs. industry
- Moving average crossovers

### Sentiment Factors

- Analyst recommendations
- News sentiment score
- Social media sentiment
- Insider buying/selling activity
- Short interest ratio
