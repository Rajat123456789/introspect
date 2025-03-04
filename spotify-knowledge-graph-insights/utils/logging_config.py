#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging configuration for the knowledge graph project.
"""

import logging
import os
import sys

def configure_logging(level=logging.INFO):
    """
    Configure logging for the application.

    Args:
        level: Logging level (default: logging.INFO)
    """
    # Set up logging format
    logging_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure the root logger
    logging.basicConfig(
        level=level,
        format=logging_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging configured with level: %s", 
                 logging.getLevelName(level))
    
    return logger

# Configure default logger
logger = configure_logging() 