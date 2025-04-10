import os
import subprocess
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Run the VaarthaAI application."""
    print("Starting VaarthaAI - Financial Assistant")

    # Check if GROQ API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY is not set in the .env file.")
        print("Some features that require GROQ API may not work correctly.")

    # Run the Streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py"])

if __name__ == "__main__":
    main()
