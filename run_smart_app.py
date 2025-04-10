import os
import subprocess
import sys
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def reset_data():
    """Reset application data if needed."""
    # Paths to check and clean if needed
    chroma_db_path = "data/chroma_db"
    
    # Check if ChromaDB directory exists and might be corrupted
    if os.path.exists(chroma_db_path):
        try:
            # Try to list files to see if there's an issue
            files = os.listdir(chroma_db_path)
            print(f"Found {len(files)} files in ChromaDB directory")
        except Exception as e:
            print(f"Error accessing ChromaDB directory: {e}")
            print("Resetting ChromaDB directory...")
            try:
                shutil.rmtree(chroma_db_path, ignore_errors=True)
                os.makedirs(chroma_db_path, exist_ok=True)
                print("ChromaDB directory has been reset")
            except Exception as e:
                print(f"Error resetting ChromaDB directory: {e}")

def main():
    """Run the VaarthaAI application with Streamlit, using the smart classifier."""
    print("Starting VaarthaAI - Financial Assistant with Smart Classifier")
    
    # Check if GROQ API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ API key not found. The smart classifier will use local methods only.")
        print("This is fine for most transactions, but complex cases might be less accurate.")
    
    # Reset data if needed
    reset_data()
    
    # Set environment variables to fix issues
    os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
    os.environ["PYTORCH_JIT"] = "0"
    
    # Disable Streamlit's file watcher to avoid PyTorch conflict
    os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Run the Streamlit app with the correct command and disable file watching
    cmd = [
        sys.executable, 
        "-m", 
        "streamlit", 
        "run", 
        "app/app.py", 
        "--server.fileWatcherType", "none",
        "--browser.serverAddress", "localhost", 
        "--server.headless", "true"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
