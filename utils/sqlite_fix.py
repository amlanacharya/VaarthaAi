"""
SQLite version fix for ChromaDB.
This module replaces the system SQLite with pysqlite3 when available.
"""
import logging

logger = logging.getLogger(__name__)

def apply_sqlite_fix():
    """
    Apply fix for SQLite version compatibility with ChromaDB.
    ChromaDB requires SQLite 3.35.0 or higher.
    """
    try:
        import sqlite3
        import re
        
        sqlite_version = sqlite3.sqlite_version
        version_numbers = re.findall(r'\d+', sqlite_version)
        
        if len(version_numbers) >= 3:
            major, minor, patch = map(int, version_numbers[:3])
            if (major, minor, patch) < (3, 35, 0):
                logger.warning(f"SQLite version {sqlite_version} is below 3.35.0 required by ChromaDB")
                try:
                    import pysqlite3
                    import sys
                    
                    logger.info("Replacing system sqlite3 with pysqlite3")
                    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
                    
                    # Verify the new version
                    import sqlite3
                    logger.info(f"Now using SQLite version: {sqlite3.sqlite_version}")
                except ImportError:
                    logger.error("pysqlite3 not available. Please install with: pip install pysqlite3-binary")
            else:
                logger.info(f"SQLite version {sqlite_version} is compatible with ChromaDB")
    except Exception as e:
        logger.error(f"Error applying SQLite fix: {e}")