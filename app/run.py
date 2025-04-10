"""
Streamlit application runner for VaarthaAI.
Handles the configuration and launch of the Streamlit web application.
"""

import os
import sys
import subprocess
import logging

from config import config
from exceptions import FileSystemError

# Configure logging
logger = logging.getLogger(__name__)


def run_streamlit(disable_watcher=False, host="localhost", port=8501, headless=False):
    """
    Run the Streamlit application.
    
    Args:
        disable_watcher: If True, disable the file watcher (fixes PyTorch conflict)
        host: Host to run the server on
        port: Port to run the server on
        headless: If True, run in headless mode
    
    Raises:
        FileSystemError: If the app file cannot be found
        RuntimeError: If Streamlit fails to start
    """
    # Check if app file exists
    app_file = os.path.join("app", "app.py")
    if not os.path.exists(app_file):
        raise FileSystemError(f"App file not found: {app_file}")
    
    # Set environment variables to fix issues
    os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
    os.environ["PYTORCH_JIT"] = "0"
    
    # Set Streamlit config environment variables
    os.environ["STREAMLIT_BROWSER_SERVER_ADDRESS"] = host
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    
    if headless:
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Set watcher environment variable if disabled
    if disable_watcher:
        os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
    
    # Build the command
    cmd = [sys.executable, "-m", "streamlit", "run", app_file]
    
    # Add extra arguments if needed
    if disable_watcher:
        cmd.extend(["--server.fileWatcherType", "none"])
    
    cmd.extend(["--browser.serverAddress", host])
    cmd.extend(["--server.port", str(port)])
    
    if headless:
        cmd.append("--server.headless")
    
    # Run Streamlit
    logger.info(f"Starting Streamlit with command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("Streamlit was stopped by the user")
    except Exception as e:
        logger.error(f"Error running Streamlit: {e}")
        raise RuntimeError(f"Failed to start Streamlit: {str(e)}")


if __name__ == "__main__":
    # Run with default settings
    run_streamlit(disable_watcher=False)