import os
import sys
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from models.transaction import Transaction, TransactionType, TransactionCategory
from models.classifier import TransactionClassifier
from utils.parser import BankStatementParser, generate_sample_transactions
from utils.database import Database

def main():
    """Run a simple test of the VaarthaAI core functionality."""
    print("\n=== VaarthaAI Simple Test ===\n")
    
    # Initialize components
    print("Initializing components...")
    db = Database()
    classifier = TransactionClassifier()
    
    # Test with sample data
    print("\nGenerating sample transactions...")
    sample_transactions = generate_sample_transactions(10)
    
    # Classify the transactions
    print("\nClassifying transactions...")
    classified_transactions = classifier.classify_batch(sample_transactions)
    
    # Display results
    print("\nClassification Results:")
    print("-" * 100)
    print(f"{'Date':<12} {'Description':<40} {'Amount':<10} {'Type':<8} {'Category':<20} {'Confidence':<10}")
    print("-" * 100)
    
    for t in classified_transactions:
        print(f"{t.date.strftime('%Y-%m-%d'):<12} {t.description[:38]:<40} {t.amount:<10.2f} {t.type.value:<8} {t.category.value:<20} {t.confidence:<10.2f}")
    
    # Save to database
    print("\nSaving transactions to database...")
    for t in classified_transactions:
        db.save_transaction(t)
    
    # Test bank statement parser
    print("\nTesting bank statement parser...")
    parser = BankStatementParser()
    
    # Parse the HDFC statement
    file_path = "data/sample_data/hdfc_coworking_statement.csv"
    print(f"Parsing file: {file_path}")
    
    try:
        batch = parser.parse(file_path, "hdfc")
        print(f"Successfully parsed {len(batch.transactions)} transactions from bank statement")
        
        # Save batch to database
        batch_id = db.save_batch(batch)
        print(f"Saved batch with ID: {batch_id}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error parsing bank statement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
