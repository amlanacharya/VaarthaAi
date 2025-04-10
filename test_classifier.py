import os
import sys
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from models.transaction import Transaction, TransactionType, TransactionCategory
from models.classifier import TransactionClassifier
from utils.parser import BankStatementParser

def test_hdfc_statement():
    """Test parsing and classification of HDFC bank statement."""
    print("\n=== Testing HDFC Bank Statement ===\n")

    # Initialize parser and classifier
    parser = BankStatementParser()
    classifier = TransactionClassifier()

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
        print("-" * 80)
        print(f"{'Date':<12} {'Description':<40} {'Amount':<10} {'Category':<20} {'Confidence':<10}")
        print("-" * 80)

        for t in classified_transactions:
            print(f"{t.date.strftime('%Y-%m-%d'):<12} {t.description[:38]:<40} {t.amount:<10.2f} {t.category.value:<20} {t.confidence:<10.2f}")

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
        return False

def test_rule_based_only():
    """Test classification using only rule-based approach (no LLM)."""
    print("\n=== Testing Rule-Based Classification Only ===\n")

    # Initialize parser and classifier with a mock GROQ client
    parser = BankStatementParser()

    # Create a classifier that will only use rule-based classification
    class RuleOnlyClassifier(TransactionClassifier):
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

        # Classify the transactions using only rules
        print("\nClassifying transactions using only rule-based approach...")
        classified_transactions = classifier.classify_batch(batch.transactions)

        # Display results
        print("\nRule-Based Classification Results:")
        print("-" * 80)
        print(f"{'Date':<12} {'Description':<40} {'Amount':<10} {'Category':<20} {'Confidence':<10}")
        print("-" * 80)

        for t in classified_transactions:
            print(f"{t.date.strftime('%Y-%m-%d'):<12} {t.description[:38]:<40} {t.amount:<10.2f} {t.category.value:<20} {t.confidence:<10.2f}")

        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Check if GROQ API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY is not set in the .env file.")
        print("The classifier will fall back to rule-based classification only.")
        test_rule_based_only()
    else:
        # Run tests with full classification
        test_hdfc_statement()
