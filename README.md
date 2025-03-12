# StockSage

Hierarchical Multi-Agent System for Top 5 US Stock Selection

## Overview

StockSage is an AI-powered investment analysis system that leverages multiple specialized agents to perform comprehensive stock analysis and selection. Using a combination of financial data analysis, sentiment analysis, and market trend detection, StockSage identifies the top 5 investment opportunities in the US stock market with detailed supporting rationales.

The system processes financial metrics, news sentiment, social media mentions, and analyst ratings to create a holistic view of each stock's potential, using a sophisticated ranking and selection methodology.

## Features

- Multi-dimensional Stock Analysis: Combines fundamental metrics, technical indicators, sentiment analysis, and macroeconomic factors
- Advanced Sentiment Analysis: Analyzes news articles, social media sentiment, and analyst opinions
- Deterministic Fallbacks: Ensures consistent operation even when API calls fail
- LangSmith Integration: Complete telemetry tracking for analysis transparency
- Customizable Analysis Parameters: Adjust analysis parameters for different investment strategies
- Streamlit UI: User-friendly interface for viewing analysis results

## Prerequisites

- Python 3.10 or higher
- Docker (optional, for containerized deployment)
- API keys:
  - Serper API (for news sentiment analysis)
  - Alpha Vantage API (for financial data)
  - LangChain API key (for agent orchestration)
  - OpenAI API key

## Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/StockSage.git
cd StockSage

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
uv sync
```

### Docker Installation

```bash
# Build Docker image
docker build -t stocksage .

# Run Docker container
docker run -p 8501:8501 \
  -e SERPER_API_KEY=your_serper_api_key \
  -e ALPHAVANTAGE_API_KEY=your_alphavantage_api_key \
  -e LANGCHAIN_API_KEY=your_langchain_api_key \
  -e LANGCHAIN_TRACING_V2=true \
  -e OPENAI_API_KEY=your_openai_api_key \
  -e LANGCHAIN_ENDPOINT=https://api.smith.langchain.com \
  -e LANGSMITH_PROJECT=stock-sage \
  stocksage
```

## Configuration

Create a .env file in the root directory with the following variables:

```bash
SERPER_API_KEY=your_serper_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=stock-sage
```

## Usage

### Running StockSage

#### CLI Mode

```bash
# Run the full analysis pipeline
python -m stocksage.main run

# Run in training mode
python -m stocksage.main train

# Test the system
python -m stocksage.main test

# Replay a previous analysis
python -m stocksage.main replay --run_id your_run_id
```

#### CrewAI Integration

```bash
from stocksage import StockSage

# Initialize and run the StockSage crew
inputs = {
    "market": "US",
    "stock_universe": "S&P 500",
    "current_year": "2025",
    "analysis_date": "2025-03-13"
}

sage = StockSage()
results = sage.crew().kickoff(inputs=inputs)
print(f"Analysis complete. Results saved to outputs directory.")
```

#### Streamlit UI

```bash
# Launch the Streamlit interface
streamlit run src/stocksage/user_interface.py
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_sentiment_analysis_tool.py

# Run with coverage report
pytest --cov=stocksage
```

## System Architecture

StockSage employs a hierarchical multi-agent architecture:

1. Data Collection Agents: Gather financial metrics, news, and market data
2. Analysis Agents: Process and interpret collected data
3. Sentiment Analysis Agent: Evaluates news sentiment and social media mentions
4. Selection & Ranking Agent: Applies scoring methodology to identify top stocks
5. Thesis Generation Agent: Creates detailed investment theses for selected stocks

## Outputs

StockSage generates the following outputs in the outputs directory:

1. Investment Recommendations JSON
   - Top 5 stock recommendations with detailed metrics
   - Complete investment thesis for each stock
   - Sentiment analysis results
   - Risk assessments

Example output structure:

```json
{
  "analysis_date": "2025-03-13",
  "investments": [
    {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "recommendation": "Buy",
      "target_price": "$210.45",
      "thesis": "Apple's strong ecosystem...",
      "sentiment_score": 0.78,
      "sentiment_label": "Bullish"
    },
    ...
  ]
}
```

2. Simplified Analysis Report

   - Condensed version of recommendations for quick review
   - Key metrics and highlights for each selected stock

3. LangSmith Run Records
   - Complete analysis traces viewable in LangSmith
   - Agent reasoning and decision processes
   - Data sources and transformation steps

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add some amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request
