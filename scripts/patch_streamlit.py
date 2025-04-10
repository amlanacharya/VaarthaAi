import os
import sys
import site
import shutil
import tempfile

def patch_streamlit():
    """
    Patch the Streamlit code to fix the PyTorch conflict.
    This modifies the local_sources_watcher.py file to handle the PyTorch error.
    """
    # Find the Streamlit installation directory
    streamlit_path = None
    for path in site.getsitepackages():
        potential_path = os.path.join(path, 'streamlit')
        if os.path.exists(potential_path):
            streamlit_path = potential_path
            break
    
    if not streamlit_path:
        print("Error: Could not find Streamlit installation directory.")
        return False
    
    # Path to the file that needs patching
    watcher_path = os.path.join(streamlit_path, 'watcher', 'local_sources_watcher.py')
    
    if not os.path.exists(watcher_path):
        print(f"Error: Could not find {watcher_path}")
        return False
    
    print(f"Found Streamlit watcher at: {watcher_path}")
    
    # Create a backup
    backup_path = watcher_path + '.bak'
    if not os.path.exists(backup_path):
        print("Creating backup of original file...")
        shutil.copy2(watcher_path, backup_path)
    
    # Read the file
    with open(watcher_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'torch._classes' in content and 'try:' in content and 'except AttributeError:' in content:
        print("File appears to be already patched.")
        return True
    
    # Find the problematic line
    target_line = 'lambda m: list(m.__path__._path),'
    if target_line not in content:
        print(f"Could not find the line to patch: {target_line}")
        return False
    
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
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        tmp.write(patched_content)
        tmp_path = tmp.name
    
    try:
        # Replace the original file with the patched version
        shutil.copy2(tmp_path, watcher_path)
        os.unlink(tmp_path)
        print("Successfully patched Streamlit to fix PyTorch conflict!")
        return True
    except Exception as e:
        print(f"Error applying patch: {e}")
        return False

if __name__ == "__main__":
    if patch_streamlit():
        print("\nPatch applied successfully. You can now run Streamlit without the PyTorch error.")
        print("To run the application, use: python run_streamlit.py")
    else:
        print("\nFailed to apply patch. Please try running with file watcher disabled:")
        print("python run_no_watcher.py")
