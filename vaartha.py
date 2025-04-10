#!/usr/bin/env python
"""
Main entry point for the VaarthaAI application.
Provides a unified interface for running all application commands.
"""

from utils.sqlite_fix import apply_sqlite_fix
apply_sqlite_fix()

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import CLI module
from cli import main

if __name__ == "__main__":
    # Run CLI
    main()