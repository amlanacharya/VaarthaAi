import os
import shutil
import sys

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def move_file(src, dest):
    """Move a file if it exists."""
    if os.path.exists(src):
        # Create destination directory if needed
        dest_dir = os.path.dirname(dest)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        # Move the file
        shutil.copy2(src, dest)
        print(f"Moved: {src} -> {dest}")
    else:
        print(f"Warning: Source file not found: {src}")

def organize_files():
    """Organize files into the proper directory structure."""
    print("Organizing files for Git...")
    
    # Create scripts directory
    create_directory("scripts")
    
    # Move utility scripts to scripts directory
    move_file("reset_app.py", "scripts/reset_app.py")
    move_file("patch_streamlit.py", "scripts/patch_streamlit.py")
    
    # Create batch directory for Windows batch files
    create_directory("batch")
    
    # Move batch files
    batch_files = [
        "run_fixed_app.bat",
        "run_smart_app.bat",
        "run_smart_classifier.bat",
        "run_smart_classifier_no_groq.bat",
        "reset_app.bat",
        "patch_streamlit.bat",
        "run_no_watcher.bat",
        "test_simple.bat"
    ]
    
    for batch_file in batch_files:
        if os.path.exists(batch_file):
            move_file(batch_file, f"batch/{batch_file}")
    
    print("\nFile organization complete!")
    print("You can now commit and push these changes to Git.")

if __name__ == "__main__":
    organize_files()
