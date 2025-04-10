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
    """Run the VaarthaAI application with Streamlit."""
    print("Starting VaarthaAI - Financial Assistant")

    # Check if GROQ API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY is not set in the .env file.")
        print("Some features that require GROQ API may not work correctly.")

    # Reset data if needed
    reset_data()

    # Set environment variables to disable warnings
    os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"

    # Fix for torch runtime error
    os.environ["PYTORCH_JIT"] = "0"

    # Run the Streamlit app with the correct command
    cmd = [sys.executable, "-m", "streamlit", "run", "app/app.py", "--browser.serverAddress", "localhost", "--server.headless", "true"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
