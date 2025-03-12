import logging
from functools import lru_cache

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@lru_cache(maxsize=128)
def get_logger():
    """
    Get a configured logger instance for the current module.

    This function returns a logger instance configured with the application's
    standard format. It uses lru_cache to ensure that only one logger instance
    is created per module name, improving performance when called frequently.

    Returns:
        logging.Logger: A configured logger instance for the current module.

    Note:
        The logger format is set at the module level with:
        "%(asctime)s - %(levelname)s - %(message)s"
        and the default logging level is INFO.

    Example:
        >>> logger = get_logger()
        >>> logger.info("Processing started")
        2025-03-12 14:30:45,123 - INFO - Processing started
    """
    return logging.getLogger(__name__)
