import os
import shutil
import sys

def reset_app_data():
    """Reset the application data by removing database files and vector stores."""
    print("This will reset all application data including:")
    print("  - Vector database (data/chroma_db)")
    print("  - SQLite database (data/vaartha.db)")
    print("  - Any temporary files")
    
    confirm = input("\nAre you sure you want to proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Reset cancelled.")
        return
    
    # Paths to clean
    paths_to_clean = [
        "data/chroma_db",
        "data/vaartha.db"
    ]
    
    # Remove each path
    for path in paths_to_clean:
        try:
            if os.path.exists(path):
                if os.path.isdir(path):
                    print(f"Removing directory: {path}")
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    print(f"Removing file: {path}")
                    os.remove(path)
        except Exception as e:
            print(f"Error removing {path}: {e}")
    
    # Recreate necessary directories
    os.makedirs("data/chroma_db", exist_ok=True)
    
    print("\nApplication data has been reset.")
    print("You can now run the application with a fresh start.")

if __name__ == "__main__":
    reset_app_data()
