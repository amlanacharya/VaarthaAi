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
from models.smart_classifier import SmartTransactionClassifier
from utils.parser import BankStatementParser

def test_hdfc_statement(use_groq=True):
    """
    Test parsing and classification of HDFC bank statement.
    
    Args:
        use_groq: Whether to use GROQ API as a fallback.
    """
    print(f"\n=== Testing HDFC Bank Statement {'with GROQ fallback' if use_groq else 'without GROQ'} ===\n")
    
    # Initialize parser and classifier
    parser = BankStatementParser()
    classifier = SmartTransactionClassifier()
    
    # If not using GROQ, disable it in the classifier
    if not use_groq:
        classifier.has_groq = False
    
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
        
        # Print statistics
        print("\nClassification Statistics:")
        print("-" * 50)
        
        # Count by confidence level
        high_confidence = sum(1 for t in classified_transactions if t.confidence >= 80)
        medium_confidence = sum(1 for t in classified_transactions if 50 <= t.confidence < 80)
        low_confidence = sum(1 for t in classified_transactions if t.confidence < 50)
        
        print(f"High confidence (>=80%): {high_confidence} transactions ({high_confidence/len(classified_transactions)*100:.1f}%)")
        print(f"Medium confidence (50-79%): {medium_confidence} transactions ({medium_confidence/len(classified_transactions)*100:.1f}%)")
        print(f"Low confidence (<50%): {low_confidence} transactions ({low_confidence/len(classified_transactions)*100:.1f}%)")
        
        # Count by category
        category_counts = {}
        for t in classified_transactions:
            category_counts[t.category] = category_counts.get(t.category, 0) + 1
        
        print("\nTransactions by Category:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{category.value:<25}: {count} transactions ({count/len(classified_transactions)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the smart transaction classifier with coworking venue data")
    parser.add_argument("--no-groq", action="store_true", help="Don't use GROQ API even if available")
    args = parser.parse_args()
    
    # Run test
    test_hdfc_statement(use_groq=not args.no_groq)
