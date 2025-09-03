"""
Logging utilities for the graph builder.
"""

import logging
import sys
from datetime import datetime


def setup_logger(name: str = "graph_builder", level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Don't add handlers if they already exist
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger


# Create default logger
logger = setup_logger()
