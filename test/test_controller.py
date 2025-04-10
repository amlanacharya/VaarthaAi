"""
Tests for the controller layer.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from models.transaction import Transaction, TransactionType, TransactionCategory
from exceptions import ParseError, DatabaseError, ClassificationError


def test_transaction_controller_process_statement(transaction_controller, mock_parser, sample_statement_file):
    """Test processing a bank statement through the transaction controller."""
    # Set up the controller with our mocks
    transaction_controller.parser = mock_parser
    
    # Process the statement
    batch, batch_id = transaction_controller.process_bank_statement(
        sample_statement_file, "hdfc"
    )
    
    # Verify the batch was processed
    assert batch is not None
    assert len(batch.transactions) > 0
    assert batch_id is not None
    
    # Verify the parser was called with correct arguments
    mock_parser.parse.assert_called_once_with(sample_statement_file, "hdfc")
    
    # Verify the classifier was called
    transaction_controller.classifier.classify_batch.assert_called_once()
    
    # Verify the batch was saved to the database
    transaction_controller.db.save_batch.assert_called_once()


def test_transaction_controller_empty_statement(transaction_controller, mock_parser):
    """Test handling an empty bank statement."""
    # Make parser return empty batch
    mock_parser.parse.return_value.transactions = []
    transaction_controller.parser = mock_parser
    
    # Process should raise an error for empty statement
    with pytest.raises(ParseError):
        transaction_controller.process_bank_statement("empty_file.csv", "hdfc")


def test_transaction_controller_handle_errors(transaction_controller, mock_parser):
    """Test error handling in the transaction controller."""
    transaction_controller.parser = mock_parser
    
    # Test parser error
    mock_parser.parse.side_effect = ParseError("Test parse error")
    with pytest.raises(ParseError):
        transaction_controller.process_bank_statement("file.csv", "hdfc")
    
    # Test classifier error
    mock_parser.parse.side_effect = None
    transaction_controller.classifier.classify_batch.side_effect = Exception("Classification failed")
    with pytest.raises(ParseError):
        transaction_controller.process_bank_statement("file.csv", "hdfc")
    
    # Test database error
    transaction_controller.classifier.classify_batch.side_effect = None
    transaction_controller.db.save_batch.side_effect = DatabaseError("Database error")
    with pytest.raises(DatabaseError):
        transaction_controller.process_bank_statement("file.csv", "hdfc")


def test_generate_sample_data(transaction_controller):
    """Test generating sample data."""
    # Mock the generate_sample_transactions function
    with patch('controllers.generate_sample_transactions') as mock_generate:
        mock_generate.return_value = [
            Transaction(
                date=datetime.now(),
                description="Test Transaction 1",
                amount=1000.0,
                type=TransactionType.DEBIT
            ),
            Transaction(
                date=datetime.now(),
                description="Test Transaction 2",
                amount=2000.0,
                type=TransactionType.CREDIT
            )
        ]
        
        # Generate sample data
        transactions = transaction_controller.generate_sample_data(count=2)
        
        # Verify the transactions were generated
        assert len(transactions) == 2
        
        # Verify the classifier was called
        transaction_controller.classifier.classify_batch.assert_called_once()
        
        # Verify the transactions were saved to the database
        assert transaction_controller.db.save_transaction.call_count == 2


def test_get_transactions(transaction_controller, sample_transaction, sample_credit_transaction):
    """Test retrieving transactions with filtering."""
    # Set up the database to return our sample transactions
    transaction_controller.db.get_transactions.return_value = [sample_transaction, sample_credit_transaction]
    
    # Get all transactions
    transactions = transaction_controller.get_transactions()
    assert len(transactions) == 2
    
    # Get transactions with category filter
    transaction_controller.get_transactions(category=TransactionCategory.EXPENSE_RENT)
    transaction_controller.db.get_transactions.assert_called_with(
        limit=100, offset=0, category=TransactionCategory.EXPENSE_RENT, transaction_type=None
    )
    
    # Get transactions with type filter
    transaction_controller.get_transactions(transaction_type=TransactionType.CREDIT)
    transaction_controller.db.get_transactions.assert_called_with(
        limit=100, offset=0, category=None, transaction_type=TransactionType.CREDIT
    )


def test_update_transaction(transaction_controller, sample_transaction):
    """Test updating a transaction."""
    # Mock db.get_transaction to return our updated transaction
    transaction_controller.db.get_transaction.return_value = sample_transaction
    
    # Update the transaction
    updated = transaction_controller.update_transaction(sample_transaction)
    
    # Verify the transaction was saved
    transaction_controller.db.save_transaction.assert_called_once_with(sample_transaction)
    
    # Verify the updated transaction was returned
    assert updated == sample_transaction


def test_insight_controller_rag_initialization(insight_controller, mock_rag):
    """Test RAG initialization in the insight controller."""
    # Initialize RAG
    insight_controller.initialize_rag()
    
    # Verify initialize_knowledge_base was called
    mock_rag.initialize_knowledge_base.assert_called_once()


def test_insight_controller_query_regulations(insight_controller, mock_rag):
    """Test querying financial regulations."""
    # Query regulations
    response = insight_controller.query_financial_regulations(
        "What are the GST requirements for small businesses?",
        regulation_type="gst"
    )
    
    # Verify RAG query was called with correct arguments
    mock_rag.query.assert_called_once_with(
        "What are the GST requirements for small businesses?",
        "gst"
    )
    
    # Verify response format
    assert "response" in response
    assert "sources" in response


def test_analyze_expense_categories(transaction_controller, sample_transaction):
    """Test analyzing expense categories."""
    # Mock get_transactions to return our sample transaction
    transaction_controller.get_transactions = MagicMock(return_value=[sample_transaction])
    
    # Analyze expenses
    result = transaction_controller.analyze_expense_categories()
    
    # Verify format of result
    assert "total_expenses" in result
    assert "category_breakdown" in result
    assert "transaction_count" in result
    
    # Verify content
    assert result["total_expenses"] == sample_transaction.amount
    assert len(result["category_breakdown"]) == 1
    assert result["category_breakdown"][0]["category"] == sample_transaction.category.value
    assert result["transaction_count"] == 1


def test_compliance_controller_gst_transactions(compliance_controller, mock_db):
    """Test retrieving GST transactions."""
    # Mock get_transactions to return different transactions for different categories
    def mock_get_transactions(limit=100, offset=0, category=None, transaction_type=None):
        if category == TransactionCategory.GST_INPUT:
            return [
                Transaction(
                    date=datetime.now(),
                    description="GST Input Credit",
                    amount=1000.0,
                    type=TransactionType.CREDIT,
                    category=TransactionCategory.GST_INPUT
                )
            ]
        elif category == TransactionCategory.GST_OUTPUT:
            return [
                Transaction(
                    date=datetime.now(),
                    description="GST Output Tax",
                    amount=2000.0,
                    type=TransactionType.DEBIT,
                    category=TransactionCategory.GST_OUTPUT
                )
            ]
        elif category == TransactionCategory.TAX_GST:
            return [
                Transaction(
                    date=datetime.now(),
                    description="GST Payment",
                    amount=1000.0,
                    type=TransactionType.DEBIT,
                    category=TransactionCategory.TAX_GST
                )
            ]
        return []
    
    mock_db.get_transactions.side_effect = mock_get_transactions
    
    # Get GST transactions
    gst_transactions = compliance_controller.get_gst_transactions()
    
    # Verify we got transactions from all GST categories
    assert len(gst_transactions) == 3
    
    # Verify each category was queried
    assert mock_db.get_transactions.call_count == 3


def test_reconcile_gst(compliance_controller):
    """Test GST reconciliation."""
    # Mock get_gst_transactions to return some transactions
    compliance_controller.get_gst_transactions = MagicMock(return_value=[
        Transaction(
            date=datetime.now(),
            description="GST Input Credit",
            amount=1000.0,
            type=TransactionType.CREDIT,
            category=TransactionCategory.GST_INPUT
        ),
        Transaction(
            date=datetime.now(),
            description="GST Output Tax",
            amount=2000.0,
            type=TransactionType.DEBIT,
            category=TransactionCategory.GST_OUTPUT
        ),
        Transaction(
            date=datetime.now(),
            description="GST Payment",
            amount=1000.0,
            type=TransactionType.DEBIT,
            category=TransactionCategory.TAX_GST
        )
    ])
    
    # Sample GST return data
    gst_return_data = {
        "input_gst": 1200.0,
        "output_gst": 2200.0,
        "gst_payment": 1000.0
    }
    
    # Reconcile GST
    result = compliance_controller.reconcile_gst(gst_return_data)
    
    # Verify format of result
    assert "bank_totals" in result
    assert "gst_totals" in result
    assert "differences" in result
    assert "transactions" in result
    
    # Verify differences calculation
    assert result["differences"]["input"] == -200.0  # 1000 - 1200
    assert result["differences"]["output"] == -200.0  # 2000 - 2200
    assert result["differences"]["payment"] == 0.0   # 1000 - 1000