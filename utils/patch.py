"""
Patch utility for VaarthaAI.
Provides functions to patch external dependencies to fix issues.
"""

import os
import sys
import site
import shutil
import tempfile
import logging

from exceptions import FileSystemError

# Configure logging
logger = logging.getLogger(__name__)


def patch_streamlit(force=False):
    """
    Patch the Streamlit code to fix the PyTorch conflict.
    This modifies the local_sources_watcher.py file to handle the PyTorch error.
    
    Args:
        force: If True, apply the patch even if it appears to be already applied
        
    Returns:
        bool: True if the patch was applied successfully, False otherwise
        
    Raises:
        FileSystemError: If the Streamlit files cannot be accessed or modified
    """
    logger.info("Patching Streamlit to fix PyTorch conflict...")
    
    # Find the Streamlit installation directory
    streamlit_path = None
    for path in site.getsitepackages():
        potential_path = os.path.join(path, 'streamlit')
        if os.path.exists(potential_path):
            streamlit_path = potential_path
            break
    
    if not streamlit_path:
        error_msg = "Could not find Streamlit installation directory."
        logger.error(error_msg)
        raise FileSystemError(error_msg)
    
    # Path to the file that needs patching
    watcher_path = os.path.join(streamlit_path, 'watcher', 'local_sources_watcher.py')
    
    if not os.path.exists(watcher_path):
        error_msg = f"Could not find {watcher_path}"
        logger.error(error_msg)
        raise FileSystemError(error_msg)
    
    logger.info(f"Found Streamlit watcher at: {watcher_path}")
    
    # Create a backup if it doesn't exist
    backup_path = watcher_path + '.bak'
    if not os.path.exists(backup_path):
        try:
            logger.info("Creating backup of original file...")
            shutil.copy2(watcher_path, backup_path)
            logger.info(f"Backup created at {backup_path}")
        except Exception as e:
            error_msg = f"Failed to create backup: {str(e)}"
            logger.error(error_msg)
            raise FileSystemError(error_msg)
    
    # Read the file
    try:
        with open(watcher_path, 'r') as f:
            content = f.read()
    except Exception as e:
        error_msg = f"Failed to read watcher file: {str(e)}"
        logger.error(error_msg)
        raise FileSystemError(error_msg)
    
    # Check if already patched
    if not force and 'try_get_path' in content and 'except AttributeError:' in content:
        logger.info("File appears to be already patched.")
        return True
    
    # Find the problematic line
    target_line = 'lambda m: list(m.__path__._path),'
    if target_line not in content:
        error_msg = f"Could not find the line to patch: {target_line}"
        logger.error(error_msg)
        raise FileSystemError(error_msg)
    
    # Create the patched version
    patched_content = content.replace(
        target_line,
        'lambda m: try_get_path(m),'
    )
    
    # Add the helper function
    helper_function = '''
def try_get_path(module):
    """Helper function to safely get module paths."""
    try:
        return list(module.__path__._path)
    except (AttributeError, RuntimeError):
        # Handle PyTorch and other modules that might cause issues
        return []
'''
    
    # Insert the helper function before the get_module_paths function
    insert_point = 'def get_module_paths(module):'
    patched_content = patched_content.replace(
        insert_point,
        helper_function + '\n' + insert_point
    )
    
    # Write the patched file
    try:
        with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
            tmp.write(patched_content)
            tmp_path = tmp.name
    except Exception as e:
        error_msg = f"Failed to create temporary file: {str(e)}"
        logger.error(error_msg)
        raise FileSystemError(error_msg)
    
    try:
        # Replace the original file with the patched version
        shutil.copy2(tmp_path, watcher_path)
        os.unlink(tmp_path)
        logger.info("Successfully patched Streamlit to fix PyTorch conflict!")
        return True
    except Exception as e:
        error_msg = f"Error applying patch: {str(e)}"
        logger.error(error_msg)
        # Try to clean up the temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass
        raise FileSystemError(error_msg)


def restore_streamlit_backup():
    """
    Restore the original Streamlit file from backup.
    
    Returns:
        bool: True if the backup was restored successfully, False otherwise
        
    Raises:
        FileSystemError: If the backup cannot be restored
    """
    logger.info("Restoring Streamlit backup...")
    
    # Find the Streamlit installation directory
    streamlit_path = None
    for path in site.getsitepackages():
        potential_path = os.path.join(path, 'streamlit')
        if os.path.exists(potential_path):
            streamlit_path = potential_path
            break
    
    if not streamlit_path:
        error_msg = "Could not find Streamlit installation directory."
        logger.error(error_msg)
        raise FileSystemError(error_msg)
    
    # Path to the file and its backup
    watcher_path = os.path.join(streamlit_path, 'watcher', 'local_sources_watcher.py')
    backup_path = watcher_path + '.bak'
    
    # Check if backup exists
    if not os.path.exists(backup_path):
        error_msg = f"Backup file not found: {backup_path}"
        logger.error(error_msg)
        raise FileSystemError(error_msg)
    
    try:
        # Restore the backup
        shutil.copy2(backup_path, watcher_path)
        logger.info("Successfully restored Streamlit backup!")
        return True
    except Exception as e:
        error_msg = f"Error restoring backup: {str(e)}"
        logger.error(error_msg)
        raise FileSystemError(error_msg)


if __name__ == "__main__":
    # If run directly, apply the patch
    try:
        if patch_streamlit():
            print("Patch applied successfully. You can now run Streamlit without the PyTorch error.")
            print("To run the application, use: python run_streamlit.py")
        else:
            print("Failed to apply patch.")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("You can try running with file watcher disabled:")
        print("python run_no_watcher.py")