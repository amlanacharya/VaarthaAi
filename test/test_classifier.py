"""
Tests for the transaction classifier.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from models.transaction import Transaction, TransactionType, TransactionCategory
from models.smart_classifier import SmartTransactionClassifier


def test_classifier_initialization():
    """Test that the classifier initializes correctly."""
    classifier = SmartTransactionClassifier(industry="coworking")
    
    assert classifier.industry == "coworking"
    assert hasattr(classifier, 'transaction_cache')
    assert isinstance(classifier.similarity_threshold, int)


def test_keyword_based_classification():
    """Test keyword-based classification."""
    classifier = SmartTransactionClassifier(industry="coworking")
    
    # Test expense classification
    transaction = Transaction(
        date=datetime.now(),
        description="Monthly office rent payment",
        amount=5000.0,
        type=TransactionType.DEBIT
    )
    
    category, confidence = classifier.keyword_based_classification(
        transaction.description, transaction.type
    )
    
    assert category == TransactionCategory.EXPENSE_RENT
    assert confidence >= 60
    
    # Test income classification
    transaction = Transaction(
        date=datetime.now(),
        description="Coworking membership payment received",
        amount=2000.0,
        type=TransactionType.CREDIT
    )
    
    category, confidence = classifier.keyword_based_classification(
        transaction.description, transaction.type
    )
    
    assert category == TransactionCategory.INCOME_BUSINESS
    assert confidence >= 60


def test_regex_based_classification():
    """Test regex-based classification."""
    classifier = SmartTransactionClassifier(industry="coworking")
    
    # Test expense classification
    transaction = Transaction(
        date=datetime.now(),
        description="RENT PAYMENT to landlord",
        amount=5000.0,
        type=TransactionType.DEBIT
    )
    
    category, confidence = classifier.regex_based_classification(
        transaction.description, transaction.type
    )
    
    assert category == TransactionCategory.EXPENSE_RENT
    assert confidence >= 80


def test_clean_description():
    """Test description cleaning."""
    classifier = SmartTransactionClassifier()
    
    # Test with special characters
    description = "RENT PAYMENT (Monthly) - #2345"
    cleaned = classifier.clean_description(description)
    
    assert cleaned == "rent payment monthly 2345"
    
    # Test with extra whitespace
    description = "  SALARY   CREDIT  "
    cleaned = classifier.clean_description(description)
    
    assert cleaned == "salary credit"


def test_transaction_cache():
    """Test the transaction cache functionality."""
    classifier = SmartTransactionClassifier()
    
    # Add a transaction to the cache
    description = "monthly rent payment"
    category = TransactionCategory.EXPENSE_RENT
    confidence = 90
    
    classifier.update_cache(description, category, confidence)
    
    # Check exact match
    cached_category, cached_confidence = classifier.check_cache(description)
    assert cached_category == category
    assert cached_confidence == 100
    
    # Check similar match
    similar_description = "monthly rent payment for office"
    cached_category, cached_confidence = classifier.check_cache(similar_description)
    
    # Should match if similarity is above threshold
    if cached_confidence >= classifier.similarity_threshold:
        assert cached_category == category
    

def test_classify_transaction():
    """Test the complete transaction classification process."""
    classifier = SmartTransactionClassifier()
    classifier.has_groq = False  # Disable GROQ for testing
    
    # Test with a clear rent expense
    transaction = Transaction(
        date=datetime.now(),
        description="Monthly Office Rent Payment",
        amount=5000.0,
        type=TransactionType.DEBIT
    )
    
    classified = classifier.classify_transaction(transaction)
    
    assert classified.category == TransactionCategory.EXPENSE_RENT
    assert classified.confidence >= 70
    
    # Test with a clear income transaction
    transaction = Transaction(
        date=datetime.now(),
        description="Coworking Membership Fee Received",
        amount=2000.0,
        type=TransactionType.CREDIT
    )
    
    classified = classifier.classify_transaction(transaction)
    
    assert classified.category == TransactionCategory.INCOME_BUSINESS
    assert classified.confidence >= 70
    
    # Test with an ambiguous transaction
    transaction = Transaction(
        date=datetime.now(),
        description="General Payment",
        amount=1000.0,
        type=TransactionType.DEBIT
    )
    
    classified = classifier.classify_transaction(transaction)
    
    # Should default to a category, even if low confidence
    assert isinstance(classified.category, TransactionCategory)
    

def test_classify_batch():
    """Test batch classification of transactions."""
    classifier = SmartTransactionClassifier()
    classifier.has_groq = False  # Disable GROQ for testing
    
    # Create a batch of transactions
    transactions = [
        Transaction(
            date=datetime.now(),
            description="Monthly Rent Payment",
            amount=5000.0,
            type=TransactionType.DEBIT
        ),
        Transaction(
            date=datetime.now(),
            description="Electricity Bill Payment",
            amount=1000.0,
            type=TransactionType.DEBIT
        ),
        Transaction(
            date=datetime.now(),
            description="Client Payment Received",
            amount=8000.0,
            type=TransactionType.CREDIT
        )
    ]
    
    # Classify the batch
    classified = classifier.classify_batch(transactions)
    
    # Check that all transactions are classified
    assert len(classified) == 3
    
    # Check classifications
    assert classified[0].category == TransactionCategory.EXPENSE_RENT
    assert classified[1].category == TransactionCategory.EXPENSE_UTILITIES
    assert classified[2].category == TransactionCategory.INCOME_BUSINESS
    
    # Check confidences
    for transaction in classified:
        assert transaction.confidence > 0


@pytest.mark.skipif("not pytest.config.getoption('--run-api')",
                   reason="Need --run-api option to run API tests")
def test_groq_based_classification():
    """Test GROQ-based classification (only runs with --run-api flag)."""
    classifier = SmartTransactionClassifier()
    
    # Skip test if no GROQ API key
    if not classifier.has_groq:
        pytest.skip("GROQ API key not available")
    
    # Test with an ambiguous transaction that would need API
    transaction = Transaction(
        date=datetime.now(),
        description="Payment to ABC Consulting for strategic advisory",
        amount=10000.0,
        type=TransactionType.DEBIT
    )
    
    with patch.object(classifier, 'keyword_based_classification', return_value=(None, 0)):
        with patch.object(classifier, 'regex_based_classification', return_value=(None, 0)):
            with patch.object(classifier, 'ml_based_classification', return_value=(None, 0)):
                with patch.object(classifier, 'bert_based_classification', return_value=(None, 0)):
                    classified = classifier.classify_transaction(transaction)
    
    # Should use GROQ API as last resort
    assert classified.category != TransactionCategory.UNCATEGORIZED
    assert classified.confidence > 0
    

def test_ml_model_training():
    """Test ML model training functionality."""
    classifier = SmartTransactionClassifier()
    
    # Sample training data
    descriptions = [
        "Monthly office rent",
        "Rent payment for office",
        "Quarterly rent invoice",
        "Internet bill payment",
        "Electricity bill",
        "Water utility payment",
        "Staff salary payment",
        "Employee payroll",
        "Consultant fee payment"
    ]
    
    categories = [
        TransactionCategory.EXPENSE_RENT,
        TransactionCategory.EXPENSE_RENT,
        TransactionCategory.EXPENSE_RENT,
        TransactionCategory.EXPENSE_UTILITIES,
        TransactionCategory.EXPENSE_UTILITIES,
        TransactionCategory.EXPENSE_UTILITIES,
        TransactionCategory.EXPENSE_SALARY,
        TransactionCategory.EXPENSE_SALARY,
        TransactionCategory.EXPENSE_PROFESSIONAL_SERVICES
    ]
    
    # Train the model
    result = classifier.train_ml_model(descriptions, categories)
    
    assert result is True
    
    # Test the trained model on new data
    test_description = "Monthly rent for office space"
    category, confidence = classifier.ml_based_classification(test_description)
    
    assert category == TransactionCategory.EXPENSE_RENT
    assert confidence > 0