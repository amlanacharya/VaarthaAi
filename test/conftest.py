"""
Pytest configuration file.
Defines fixtures and test setup for VaarthaAI tests.
"""

import os
import sys
import tempfile
import pytest
from datetime import datetime
import sqlite3
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from config import get_config
from models.transaction import Transaction, TransactionType, TransactionCategory, TransactionBatch
from models.smart_classifier import SmartTransactionClassifier
from models.rag import FinancialRAG
from utils.parser import BankStatementParser
from utils.database import Database
from controllers import TransactionController, InsightController, ComplianceController


# Set test environment
os.environ["VAARTHA_ENV"] = "testing"
config = get_config()


@pytest.fixture
def sample_transaction():
    """Fixture for a sample transaction."""
    return Transaction(
        id="test-transaction-id",
        date=datetime.now(),
        description="Test Transaction Description",
        amount=1000.0,
        type=TransactionType.DEBIT,
        category=TransactionCategory.EXPENSE_RENT,
        subcategory="Office Rent",
        notes="Test transaction for unit tests",
        source="Test",
        confidence=90.0,
        metadata={"test_key": "test_value"}
    )


@pytest.fixture
def sample_credit_transaction():
    """Fixture for a sample credit transaction."""
    return Transaction(
        id="test-credit-transaction-id",
        date=datetime.now(),
        description="Salary Payment",
        amount=5000.0,
        type=TransactionType.CREDIT,
        category=TransactionCategory.INCOME_BUSINESS,
        confidence=85.0,
        source="Test"
    )


@pytest.fixture
def sample_transaction_batch(sample_transaction, sample_credit_transaction):
    """Fixture for a sample transaction batch."""
    return TransactionBatch(
        transactions=[sample_transaction, sample_credit_transaction],
        source="Test Batch",
        metadata={"test_key": "test_value"}
    )


@pytest.fixture
def sample_statement_file():
    """Fixture for a sample bank statement file."""
    # Create a temporary file with sample statement data
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp.write(b"""Date,Narration,Chq/Ref No,Value Date,Withdrawal Amt,Deposit Amt,Closing Balance
01/04/2023,Opening Balance,,,,,10000.00
02/04/2023,SALARY CREDIT,NEFT123456,02/04/2023,,5000.00,15000.00
05/04/2023,RENT PAYMENT,NEFT789012,05/04/2023,2000.00,,13000.00
10/04/2023,ELECTRICITY BILL,BILL123,10/04/2023,500.00,,12500.00
15/04/2023,CLIENT PAYMENT,IMPS456789,15/04/2023,,3000.00,15500.00
""")
        tmp_path = tmp.name
    
    # Return the path and ensure it's deleted after the test
    yield tmp_path
    os.unlink(tmp_path)


@pytest.fixture
def mock_classifier():
    """Fixture for a mock transaction classifier."""
    classifier = MagicMock(spec=SmartTransactionClassifier)
    
    # Make the classifier.classify_transaction method return the transaction with a category
    def classify_transaction(transaction):
        # Set a category based on the description
        if "RENT" in transaction.description:
            transaction.category = TransactionCategory.EXPENSE_RENT
        elif "SALARY" in transaction.description:
            transaction.category = TransactionCategory.INCOME_BUSINESS
        elif "ELECTRICITY" in transaction.description:
            transaction.category = TransactionCategory.EXPENSE_UTILITIES
        elif "CLIENT" in transaction.description:
            transaction.category = TransactionCategory.INCOME_BUSINESS
        else:
            transaction.category = TransactionCategory.UNCATEGORIZED
        
        transaction.confidence = 90.0
        return transaction
    
    # Make the classifier.classify_batch method use classify_transaction
    def classify_batch(transactions):
        return [classify_transaction(t) for t in transactions]
    
    classifier.classify_transaction.side_effect = classify_transaction
    classifier.classify_batch.side_effect = classify_batch
    
    return classifier


@pytest.fixture
def mock_parser():
    """Fixture for a mock bank statement parser."""
    parser = MagicMock(spec=BankStatementParser)
    
    def parse(file_path, bank_name):
        # Return a sample batch with different transactions depending on the bank
        transactions = []
        
        if bank_name.lower() == "hdfc":
            transactions = [
                Transaction(
                    date=datetime(2023, 4, 2),
                    description="SALARY CREDIT",
                    amount=5000.0,
                    type=TransactionType.CREDIT,
                    source="HDFC Bank Statement"
                ),
                Transaction(
                    date=datetime(2023, 4, 5),
                    description="RENT PAYMENT",
                    amount=2000.0,
                    type=TransactionType.DEBIT,
                    source="HDFC Bank Statement"
                )
            ]
        elif bank_name.lower() == "sbi":
            transactions = [
                Transaction(
                    date=datetime(2023, 4, 10),
                    description="ELECTRICITY BILL",
                    amount=500.0,
                    type=TransactionType.DEBIT,
                    source="SBI Bank Statement"
                ),
                Transaction(
                    date=datetime(2023, 4, 15),
                    description="CLIENT PAYMENT",
                    amount=3000.0,
                    type=TransactionType.CREDIT,
                    source="SBI Bank Statement"
                )
            ]
        
        return TransactionBatch(
            transactions=transactions,
            source=f"{bank_name.upper()} Bank Statement",
            metadata={"file_path": file_path}
        )
    
    parser.parse.side_effect = parse
    
    return parser


@pytest.fixture
def mock_db():
    """Fixture for a mock database."""
    db = MagicMock(spec=Database)
    
    # Dictionary to store transactions
    transactions_store = {}
    
    # Dictionary to store batches
    batches_store = {}
    
    # Method to save a transaction
    def save_transaction(transaction):
        # Generate ID if not present
        if transaction.id is None:
            import uuid
            transaction.id = str(uuid.uuid4())
        
        # Store the transaction
        transactions_store[transaction.id] = transaction
        return transaction.id
    
    # Method to save a batch
    def save_batch(batch):
        # Generate batch ID
        import uuid
        batch_id = str(uuid.uuid4())
        
        # Store the batch
        batches_store[batch_id] = batch
        
        # Save each transaction
        for transaction in batch.transactions:
            save_transaction(transaction)
        
        return batch_id
    
    # Method to get a transaction
    def get_transaction(transaction_id):
        return transactions_store.get(transaction_id)
    
    # Method to get transactions
    def get_transactions(limit=100, offset=0, category=None, transaction_type=None):
        transactions = list(transactions_store.values())
        
        # Apply filters
        if category:
            category_value = category.value if hasattr(category, 'value') else category
            transactions = [t for t in transactions if t.category.value == category_value]
        
        if transaction_type:
            type_value = transaction_type.value if hasattr(transaction_type, 'value') else transaction_type
            transactions = [t for t in transactions if t.type.value == type_value]
        
        # Apply limit and offset
        return transactions[offset:offset+limit]
    
    # Method to get a batch
    def get_batch(batch_id):
        return batches_store.get(batch_id)
    
    # Method to get batches
    def get_batches(limit=10, offset=0):
        batches = list(batches_store.values())
        
        # Convert to summaries
        summaries = []
        for i, batch in enumerate(batches[offset:offset+limit]):
            summaries.append({
                "id": f"batch-{i}",
                "source": batch.source,
                "created_at": datetime.now().isoformat(),
                "transaction_count": len(batch.transactions)
            })
        
        return summaries
    
    # Set up the mock methods
    db.save_transaction.side_effect = save_transaction
    db.save_batch.side_effect = save_batch
    db.get_transaction.side_effect = get_transaction
    db.get_transactions.side_effect = get_transactions
    db.get_batch.side_effect = get_batch
    db.get_batches.side_effect = get_batches
    
    return db


@pytest.fixture
def mock_rag():
    """Fixture for a mock RAG system."""
    rag = MagicMock(spec=FinancialRAG)
    
    def query(query_text, regulation_type=None):
        # Return a sample response
        return {
            "response": f"This is a response to the query: {query_text}",
            "sources": [
                {
                    "title": "Sample Regulation",
                    "section": "Section 1",
                    "section_title": "Introduction",
                    "regulation_type": regulation_type or "general",
                    "source": "sample_regulation.json"
                }
            ]
        }
    
    rag.query.side_effect = query
    
    return rag


@pytest.fixture
def transaction_controller(mock_db, mock_classifier):
    """Fixture for a transaction controller with mocked dependencies."""
    return TransactionController(db=mock_db, classifier=mock_classifier)


@pytest.fixture
def insight_controller(mock_db, mock_rag):
    """Fixture for an insight controller with mocked dependencies."""
    return InsightController(db=mock_db, rag=mock_rag)


@pytest.fixture
def compliance_controller(mock_db):
    """Fixture for a compliance controller with mocked dependencies."""
    return ComplianceController(db=mock_db)


@pytest.fixture
def in_memory_db():
    """Fixture for a real in-memory SQLite database."""
    # Create a temporary directory for the database file
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.db")
        
        # Initialize the database
        db = Database(db_path=db_path)
        
        yield db


@pytest.fixture
def real_classifier():
    """Fixture for a real transaction classifier without API access."""
    classifier = SmartTransactionClassifier(industry="general")
    classifier.has_groq = False  # Disable API access for tests
    
    return classifier