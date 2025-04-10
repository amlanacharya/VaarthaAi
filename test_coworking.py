import os
import sys
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import argparse

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from models.transaction import Transaction, TransactionType, TransactionCategory
from models.classifier import TransactionClassifier
from utils.parser import BankStatementParser

def test_hdfc_statement(use_llm=True):
    """
    Test parsing and classification of HDFC bank statement.
    
    Args:
        use_llm: Whether to use LLM for classification or just rule-based.
    """
    print(f"\n=== Testing HDFC Bank Statement {'with LLM' if use_llm else 'with rules only'} ===\n")
    
    # Initialize parser and classifier
    parser = BankStatementParser()
    
    if use_llm:
        classifier = TransactionClassifier()
    else:
        # Create a classifier that will only use rule-based classification
        class RuleOnlyClassifier(TransactionClassifier):
            def __init__(self, industry="general"):
                super().__init__(industry)
                self.has_llm = False
                
            def classify_with_llm(self, transaction):
                # Skip LLM classification and return uncategorized with low confidence
                return (TransactionCategory.UNCATEGORIZED, 0.0)
        
        classifier = RuleOnlyClassifier()
    
    # Parse the statement
    file_path = "data/sample_data/hdfc_coworking_statement.csv"
    print(f"Parsing file: {file_path}")
    
    try:
        batch = parser.parse(file_path, "hdfc")
        print(f"Successfully parsed {len(batch.transactions)} transactions")
        
        # Classify the transactions
        print("\nClassifying transactions...")
        classified_transactions = classifier.classify_batch(batch.transactions)
        
        # Display results
        print("\nClassification Results:")
        print("-" * 100)
        print(f"{'Date':<12} {'Description':<40} {'Amount':<10} {'Type':<8} {'Category':<20} {'Confidence':<10}")
        print("-" * 100)
        
        for t in classified_transactions:
            print(f"{t.date.strftime('%Y-%m-%d'):<12} {t.description[:38]:<40} {t.amount:<10.2f} {t.type.value:<8} {t.category.value:<20} {t.confidence:<10.2f}")
        
        # Summary by category
        print("\nCategory Summary:")
        print("-" * 50)
        
        category_totals = {}
        for t in classified_transactions:
            if t.category not in category_totals:
                category_totals[t.category] = 0
            
            # Add or subtract based on transaction type
            if t.type == TransactionType.CREDIT:
                category_totals[t.category] += t.amount
            else:
                category_totals[t.category] -= t.amount
        
        for category, amount in sorted(category_totals.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"{category.value:<25}: ₹{amount:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the transaction classifier with coworking venue data")
    parser.add_argument("--rules-only", action="store_true", help="Use only rule-based classification (no LLM)")
    args = parser.parse_args()
    
    # Check if GROQ API key is set
    if not os.getenv("GROQ_API_KEY") and not args.rules_only:
        print("Warning: GROQ_API_KEY is not set in the .env file.")
        print("Falling back to rule-based classification only.")
        use_llm = False
    else:
        use_llm = not args.rules_only
    
    # Run test
    test_hdfc_statement(use_llm=use_llm)
