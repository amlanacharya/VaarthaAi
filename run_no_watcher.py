import os
import subprocess
import sys
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Run the VaarthaAI application with Streamlit, disabling the file watcher."""
    print("Starting VaarthaAI - Financial Assistant (with file watcher disabled)")
    
    # Check if GROQ API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY is not set in the .env file.")
        print("Some features that require GROQ API may not work correctly.")
    
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
