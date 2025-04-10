"""
Command Line Interface for VaarthaAI.
Provides a unified CLI interface for all application functions.
"""

import argparse
import sys
import os
import logging
from typing import List, Optional

from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args(args: Optional[List[str]] = None):
    """
    Parse command line arguments.
    
    Args:
        args: Command line arguments (if None, uses sys.argv)
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="VaarthaAI Financial Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run app command
    app_parser = subparsers.add_parser("run", help="Run the web application")
    app_parser.add_argument("--no-watcher", action="store_true", help="Disable file watcher (fixes PyTorch conflict)")
    app_parser.add_argument("--host", default="localhost", help="Host to run the server on")
    app_parser.add_argument("--port", type=int, default=8501, help="Port to run the server on")
    app_parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    
    # Test classifier command
    test_parser = subparsers.add_parser("test", help="Test the classifier")
    test_parser.add_argument("--no-groq", action="store_true", help="Don't use GROQ API even if available")
    test_parser.add_argument("--file", default="data/sample_data/hdfc_coworking_statement.csv", 
                            help="Path to test file")
    test_parser.add_argument("--bank", default="hdfc", help="Bank name for the test file")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset application data")
    reset_parser.add_argument("--force", action="store_true", help="Force reset without confirmation")
    reset_parser.add_argument("--database", action="store_true", help="Reset only the database")
    reset_parser.add_argument("--vector-db", action="store_true", help="Reset only the vector database")
    
    # Patch command
    patch_parser = subparsers.add_parser("patch", help="Apply patches to fix issues")
    patch_parser.add_argument("--streamlit", action="store_true", help="Patch Streamlit to fix PyTorch conflict")
    patch_parser.add_argument("--force", action="store_true", help="Force reapplication of patches")
    
    # Generate sample data command
    sample_parser = subparsers.add_parser("sample", help="Generate sample data")
    sample_parser.add_argument("--count", type=int, default=20, help="Number of transactions to generate")
    sample_parser.add_argument("--industry", default="coworking", help="Industry context for sample data")
    
    return parser.parse_args(args)


def run_app(args):
    """
    Run the web application.
    
    Args:
        args: Parsed command line arguments
    """
    from app.run import run_streamlit
    
    # Set watcher option
    disable_watcher = args.no_watcher
    host = args.host
    port = args.port
    headless = args.headless
    
    run_streamlit(disable_watcher=disable_watcher, host=host, port=port, headless=headless)


def test_classifier(args):
    """
    Test the transaction classifier.
    
    Args:
        args: Parsed command line arguments
    """
    from test.test_classifier import test_hdfc_statement
    
    # Get arguments
    use_groq = not args.no_groq
    file_path = args.file
    bank_name = args.bank
    
    # Check if GROQ API key is available
    if use_groq and not config.GROQ_API_KEY:
        logger.warning("GROQ API key not found. Running test without GROQ.")
        use_groq = False
    
    # Run test
    test_hdfc_statement(use_groq=use_groq, file_path=file_path, bank_name=bank_name)


def reset_app_data(args):
    """
    Reset application data.
    
    Args:
        args: Parsed command line arguments
    """
    from utils.reset import reset_app_data
    
    # Get arguments
    force = args.force
    reset_db = args.database
    reset_vector_db = args.vector_db
    
    # If neither specific reset is selected, reset everything
    if not reset_db and not reset_vector_db:
        reset_db = True
        reset_vector_db = True
    
    # Confirm reset if not forced
    if not force:
        if reset_db and reset_vector_db:
            target = "all application data"
        elif reset_db:
            target = "the database"
        else:
            target = "the vector database"
            
        confirm = input(f"Are you sure you want to reset {target}? (y/n): ")
        if confirm.lower() != 'y':
            logger.info("Reset cancelled.")
            return
    
    # Perform reset
    reset_app_data(reset_db=reset_db, reset_vector_db=reset_vector_db)


def patch_streamlit(args):
    """
    Apply patches to fix issues.
    
    Args:
        args: Parsed command line arguments
    """
    from utils.patch import patch_streamlit
    
    # Get arguments
    force = args.force
    
    # Apply patch
    if args.streamlit:
        patch_streamlit(force=force)


def generate_sample_data(args):
    """
    Generate sample data.
    
    Args:
        args: Parsed command line arguments
    """
    from controllers import TransactionController
    
    # Get arguments
    count = args.count
    industry = args.industry
    
    # Generate sample data
    controller = TransactionController()
    transactions = controller.generate_sample_data(count=count)
    
    logger.info(f"Generated {len(transactions)} sample transactions")
    
    # Print summary
    credit_count = sum(1 for t in transactions if t.type.value == "credit")
    debit_count = sum(1 for t in transactions if t.type.value == "debit")
    
    logger.info(f"Credit transactions: {credit_count}")
    logger.info(f"Debit transactions: {debit_count}")


def main(args=None):
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments (if None, uses sys.argv)
    """
    args = parse_args(args)
    
    # Check command and execute appropriate function
    if args.command == "run":
        run_app(args)
    elif args.command == "test":
        test_classifier(args)
    elif args.command == "reset":
        reset_app_data(args)
    elif args.command == "patch":
        patch_streamlit(args)
    elif args.command == "sample":
        generate_sample_data(args)
    else:
        # If no command is specified, show help
        parse_args(["--help"])


if __name__ == "__main__":
    main()