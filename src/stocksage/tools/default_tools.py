from crewai_tools import (
    FirecrawlCrawlWebsiteTool,
    FirecrawlScrapeWebsiteTool,
)
import os
from dotenv import load_dotenv

load_dotenv()


def get_firecrawl_crawl_website_tool() -> FirecrawlCrawlWebsiteTool:
    """
    Create and return a configured FirecrawlCrawlWebsiteTool instance.

    This function initializes a FirecrawlCrawlWebsiteTool with the API key
    from environment variables. The tool is used to crawl websites and extract
    structured information from multiple pages, following links within the
    domain.

    Returns:
        FirecrawlCrawlWebsiteTool: A configured web crawler tool ready to use
            with the FireCrawl service API.

    Note:
        Requires the FIRECRAWL_API_KEY environment variable to be set.
        Uses the load_dotenv() call at module level to load environment variables.
    """
    return FirecrawlCrawlWebsiteTool(api_key=os.getenv("FIRECRAWL_API_KEY"))


def get_firecrawl_scrape_website_tool() -> FirecrawlScrapeWebsiteTool:
    """
    Create and return a configured FirecrawlScrapeWebsiteTool instance.

    This function initializes a FirecrawlScrapeWebsiteTool with the API key
    from environment variables. The tool is used to scrape content from a
    specific webpage and extract structured information from it.

    Returns:
        FirecrawlScrapeWebsiteTool: A configured web scraping tool ready to use
            with the FireCrawl service API.

    Note:
        Requires the FIRECRAWL_API_KEY environment variable to be set.
        Uses the load_dotenv() call at module level to load environment variables.
    """
    return FirecrawlScrapeWebsiteTool(api_key=os.getenv("FIRECRAWL_API_KEY"))
