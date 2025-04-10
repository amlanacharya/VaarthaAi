"""
Data reset utility for VaarthaAI.
Provides functions to reset application data and databases.
"""

import os
import shutil
import logging

from config import config
from exceptions import FileSystemError

# Configure logging
logger = logging.getLogger(__name__)


def reset_app_data(reset_db=True, reset_vector_db=True):
    """
    Reset application data by removing database files and vector stores.
    
    Args:
        reset_db: If True, reset the SQLite database
        reset_vector_db: If True, reset the vector database (ChromaDB)
    
    Raises:
        FileSystemError: If the data directory cannot be accessed or modified
    """
    logger.info("Resetting application data...")
    
    # Paths to clean
    paths_to_clean = []
    
    if reset_db:
        db_path = config.DATABASE_URL
        if db_path.startswith("sqlite:///"):
            db_path = db_path[10:]
        paths_to_clean.append(db_path)
        logger.info(f"Will reset database at {db_path}")
    
    if reset_vector_db:
        vector_db_path = config.CHROMA_DB_PATH
        paths_to_clean.append(vector_db_path)
        logger.info(f"Will reset vector database at {vector_db_path}")
    
    if not paths_to_clean:
        logger.warning("No data paths selected for reset")
        return
    
    # Remove each path
    for path in paths_to_clean:
        try:
            if os.path.exists(path):
                if os.path.isdir(path):
                    logger.info(f"Removing directory: {path}")
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    logger.info(f"Removing file: {path}")
                    os.remove(path)
        except Exception as e:
            logger.error(f"Error removing {path}: {e}")
            raise FileSystemError(f"Failed to remove {path}: {str(e)}")
    
    # Recreate necessary directories
    if reset_vector_db:
        try:
            os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
            logger.info(f"Recreated directory: {config.CHROMA_DB_PATH}")
        except Exception as e:
            logger.error(f"Error recreating {config.CHROMA_DB_PATH}: {e}")
            raise FileSystemError(f"Failed to recreate {config.CHROMA_DB_PATH}: {str(e)}")
    
    # Create data directory if it doesn't exist
    try:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        logger.info(f"Ensured data directory exists: {config.DATA_DIR}")
    except Exception as e:
        logger.error(f"Error creating {config.DATA_DIR}: {e}")
        raise FileSystemError(f"Failed to create {config.DATA_DIR}: {str(e)}")
    
    logger.info("Application data has been reset")


def clean_temp_files():
    """
    Clean temporary files created by the application.
    
    Raises:
        FileSystemError: If the temp files cannot be removed
    """
    logger.info("Cleaning temporary files...")
    
    # Temporary directories to clean
    temp_dirs = [
        os.path.join("data", "temp"),
        "temp"
    ]
    
    # Temporary file patterns to clean
    temp_patterns = [
        "*.tmp",
        "*.temp",
        "temp_*.csv",
        "temp_*.json"
    ]
    
    # Remove temp directories
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error removing {temp_dir}: {e}")
                raise FileSystemError(f"Failed to remove {temp_dir}: {str(e)}")
    
    # Remove temp files from data directory
    import glob
    try:
        for pattern in temp_patterns:
            for file_path in glob.glob(os.path.join(config.DATA_DIR, pattern)):
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error removing temporary files: {e}")
        raise FileSystemError(f"Failed to remove temporary files: {str(e)}")
    
    logger.info("Temporary files have been cleaned")


if __name__ == "__main__":
    # If run directly, ask for confirmation
    confirm = input("This will reset all application data. Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        # Reset all data
        reset_app_data(reset_db=True, reset_vector_db=True)
        # Clean temp files
        clean_temp_files()
        print("Application data has been reset.")
    else:
        print("Reset cancelled.")