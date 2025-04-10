"""
Controller layer for VaarthaAI.
Implements business logic and separates it from presentation layer.
"""

import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from config import config
from exceptions import ParseError, ClassificationError, DatabaseError
from models.transaction import Transaction, TransactionBatch, TransactionType, TransactionCategory
from models.smart_classifier import SmartTransactionClassifier
from models.rag import FinancialRAG
from utils.parser import BankStatementParser, generate_sample_transactions
from utils.database import Database

# Configure logging
logger = logging.getLogger(__name__)


class TransactionController:
    """
    Controller for transaction-related operations.
    Manages transaction processing, classification, and storage.
    """
    
    def __init__(self, db: Optional[Database] = None, classifier: Optional[SmartTransactionClassifier] = None):
        """
        Initialize the controller with dependencies.
        
        Args:
            db: Database instance for storage (injected for testability)
            classifier: Classifier instance for transaction classification
        """
        self.db = db or Database()
        self.classifier = classifier or SmartTransactionClassifier(industry=config.DEFAULT_INDUSTRY)
        self.parser = BankStatementParser()
    
    def process_bank_statement(self, file_path: str, bank_name: str) -> Tuple[TransactionBatch, str]:
        """
        Process a bank statement and return classified transactions.
        
        Args:
            file_path: Path to the bank statement file
            bank_name: Name of the bank
            
        Returns:
            Tuple containing the processed batch and its ID
            
        Raises:
            ParseError: If statement parsing fails
            ClassificationError: If classification fails
            DatabaseError: If database operations fail
        """
        try:
            # Parse the bank statement
            batch = self.parser.parse(file_path, bank_name.lower())
            
            if not batch.transactions:
                raise ParseError(f"No transactions found in the statement from {bank_name}")
            
            # Classify the transactions
            batch.transactions = self.classifier.classify_batch(batch.transactions)
            
            # Save to database
            batch_id = self.db.save_batch(batch)
            
            logger.info(f"Successfully processed {len(batch.transactions)} transactions from {bank_name}")
            return batch, batch_id
            
        except ParseError as e:
            logger.error(f"Error parsing statement: {e}")
            raise
        except ClassificationError as e:
            logger.error(f"Error classifying transactions: {e}")
            raise
        except DatabaseError as e:
            logger.error(f"Error saving to database: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing statement: {e}")
            raise ParseError(f"Failed to process bank statement: {str(e)}")
    
    def generate_sample_data(self, count: int = 20) -> List[Transaction]:
        """
        Generate and process sample transactions.
        
        Args:
            count: Number of transactions to generate
            
        Returns:
            List of classified transactions
            
        Raises:
            ClassificationError: If classification fails
            DatabaseError: If database operations fail
        """
        try:
            # Generate sample transactions
            sample_transactions = generate_sample_transactions(count)
            
            # Classify the transactions
            classified_transactions = self.classifier.classify_batch(sample_transactions)
            
            # Save to database
            for transaction in classified_transactions:
                self.db.save_transaction(transaction)
            
            logger.info(f"Generated and classified {count} sample transactions")
            return classified_transactions
            
        except ClassificationError as e:
            logger.error(f"Error classifying sample transactions: {e}")
            raise
        except DatabaseError as e:
            logger.error(f"Error saving sample transactions: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating sample data: {e}")
            raise ClassificationError(f"Failed to generate sample data: {str(e)}")
    
    def get_transactions(self, limit: int = 100, offset: int = 0, 
                          category: Optional[TransactionCategory] = None,
                          transaction_type: Optional[TransactionType] = None) -> List[Transaction]:
        """
        Get transactions with optional filtering.
        
        Args:
            limit: Maximum number of transactions to return
            offset: Number of transactions to skip
            category: Filter by category
            transaction_type: Filter by transaction type
            
        Returns:
            List of transactions
            
        Raises:
            DatabaseError: If database operations fail
        """
        try:
            return self.db.get_transactions(
                limit=limit, 
                offset=offset, 
                category=category, 
                transaction_type=transaction_type
            )
        except DatabaseError as e:
            logger.error(f"Error retrieving transactions: {e}")
            raise
    
    def update_transaction(self, transaction: Transaction) -> Transaction:
        """
        Update a transaction in the database.
        
        Args:
            transaction: Transaction to update
            
        Returns:
            Updated transaction
            
        Raises:
            DatabaseError: If database operations fail
        """
        try:
            self.db.save_transaction(transaction)
            return self.db.get_transaction(transaction.id)
        except DatabaseError as e:
            logger.error(f"Error updating transaction: {e}")
            raise
    
    def get_category_summary(self, transaction_type: Optional[TransactionType] = None) -> Dict[TransactionCategory, float]:
        """
        Get summary of transactions by category.
        
        Args:
            transaction_type: Filter by transaction type
            
        Returns:
            Dictionary mapping categories to total amounts
            
        Raises:
            DatabaseError: If database operations fail
        """
        try:
            transactions = self.get_transactions(limit=1000, transaction_type=transaction_type)
            
            category_totals = {}
            for t in transactions:
                if t.category not in category_totals:
                    category_totals[t.category] = 0.0
                category_totals[t.category] += t.amount
            
            return category_totals
            
        except DatabaseError as e:
            logger.error(f"Error generating category summary: {e}")
            raise


class InsightController:
    """
    Controller for financial insights and analysis.
    Manages RAG queries and financial analysis.
    """
    
    def __init__(self, db: Optional[Database] = None, rag: Optional[FinancialRAG] = None):
        """
        Initialize the controller with dependencies.
        
        Args:
            db: Database instance for storage (injected for testability)
            rag: RAG instance for financial queries
        """
        self.db = db or Database()
        self.rag = rag or FinancialRAG()
        
        # Initialize RAG knowledge base if needed
        if config.ENABLE_RAG:
            self.initialize_rag()
    
    def initialize_rag(self):
        """Initialize the RAG knowledge base."""
        try:
            self.rag.initialize_knowledge_base()
            logger.info("RAG knowledge base initialized")
        except Exception as e:
            logger.error(f"Error initializing RAG knowledge base: {e}")
    
    def query_financial_regulations(self, query: str, regulation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Query financial regulations using RAG.
        
        Args:
            query: The query string
            regulation_type: Optional filter for regulation type
            
        Returns:
            Dictionary with response and sources
            
        Raises:
            RAGError: If RAG query fails
        """
        try:
            return self.rag.query(query, regulation_type)
        except Exception as e:
            logger.error(f"Error querying financial regulations: {e}")
            raise
    
    def analyze_expense_categories(self) -> Dict[str, Any]:
        """
        Analyze expenses by category.
        
        Returns:
            Dictionary with expense analysis
            
        Raises:
            DatabaseError: If database operations fail
        """
        try:
            # Get all expense transactions
            transactions = self.db.get_transactions(
                limit=1000, 
                transaction_type=TransactionType.DEBIT
            )
            
            # Organize by category
            category_totals = {}
            for t in transactions:
                category = t.category.value
                if category not in category_totals:
                    category_totals[category] = 0.0
                category_totals[category] += t.amount
            
            # Sort by amount (descending)
            sorted_categories = [
                {"category": category, "amount": amount}
                for category, amount in sorted(
                    category_totals.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
            ]
            
            return {
                "total_expenses": sum(category_totals.values()),
                "category_breakdown": sorted_categories,
                "transaction_count": len(transactions)
            }
            
        except DatabaseError as e:
            logger.error(f"Error analyzing expense categories: {e}")
            raise
    
    def find_potential_deductions(self) -> List[Transaction]:
        """
        Find potential tax deductions in transactions.
        
        Returns:
            List of transactions that might be tax deductible
            
        Raises:
            DatabaseError: If database operations fail
        """
        try:
            # Categories that are typically tax deductible
            deductible_categories = [
                TransactionCategory.EXPENSE_RENT,
                TransactionCategory.EXPENSE_INSURANCE,
                TransactionCategory.EXPENSE_PROFESSIONAL_SERVICES,
                TransactionCategory.EXPENSE_OFFICE_SUPPLIES,
                TransactionCategory.EXPENSE_MAINTENANCE
            ]
            
            potential_deductions = []
            
            # Check each category
            for category in deductible_categories:
                transactions = self.db.get_transactions(
                    limit=100,
                    transaction_type=TransactionType.DEBIT,
                    category=category
                )
                potential_deductions.extend(transactions)
            
            return potential_deductions
            
        except DatabaseError as e:
            logger.error(f"Error finding potential deductions: {e}")
            raise
    
    def generate_monthly_summary(self) -> List[Dict[str, Any]]:
        """
        Generate monthly financial summary.
        
        Returns:
            List of monthly summaries
            
        Raises:
            DatabaseError: If database operations fail
        """
        try:
            # Get all transactions
            transactions = self.db.get_transactions(limit=5000)
            
            # Group by month
            monthly_data = {}
            
            for t in transactions:
                month_key = t.date.strftime('%Y-%m')
                
                if month_key not in monthly_data:
                    monthly_data[month_key] = {
                        "month": t.date.strftime('%B %Y'),
                        "income": 0.0,
                        "expenses": 0.0,
                        "net": 0.0
                    }
                
                if t.type == TransactionType.CREDIT:
                    monthly_data[month_key]["income"] += t.amount
                else:
                    monthly_data[month_key]["expenses"] += t.amount
            
            # Calculate net amounts
            for month_key in monthly_data:
                monthly_data[month_key]["net"] = (
                    monthly_data[month_key]["income"] - 
                    monthly_data[month_key]["expenses"]
                )
            
            # Convert to sorted list
            sorted_months = sorted(monthly_data.values(), key=lambda x: x["month"])
            
            return sorted_months
            
        except DatabaseError as e:
            logger.error(f"Error generating monthly summary: {e}")
            raise


class ComplianceController:
    """
    Controller for compliance-related operations.
    Manages GST reconciliation and tax deduction finding.
    """
    
    def __init__(self, db: Optional[Database] = None):
        """
        Initialize the controller with dependencies.
        
        Args:
            db: Database instance for storage (injected for testability)
        """
        self.db = db or Database()
    
    def get_gst_transactions(self) -> List[Transaction]:
        """
        Get GST-related transactions.
        
        Returns:
            List of GST-related transactions
            
        Raises:
            DatabaseError: If database operations fail
        """
        try:
            # GST-related categories
            gst_categories = [
                TransactionCategory.GST_INPUT,
                TransactionCategory.GST_OUTPUT,
                TransactionCategory.TAX_GST
            ]
            
            gst_transactions = []
            
            # Fetch transactions for each GST category
            for category in gst_categories:
                transactions = self.db.get_transactions(
                    limit=100,
                    category=category
                )
                gst_transactions.extend(transactions)
            
            return gst_transactions
            
        except DatabaseError as e:
            logger.error(f"Error retrieving GST transactions: {e}")
            raise
    
    def reconcile_gst(self, gst_return_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconcile bank transactions with GST return data.
        
        Args:
            gst_return_data: Data from GST return
            
        Returns:
            Reconciliation results
            
        Raises:
            DatabaseError: If database operations fail
        """
        try:
            # Get GST transactions
            transactions = self.get_gst_transactions()
            
            # Calculate totals from transactions
            bank_totals = {
                "input": sum(t.amount for t in transactions if t.category == TransactionCategory.GST_INPUT),
                "output": sum(t.amount for t in transactions if t.category == TransactionCategory.GST_OUTPUT),
                "payment": sum(t.amount for t in transactions if t.category == TransactionCategory.TAX_GST)
            }
            
            # Get totals from GST return
            gst_totals = {
                "input": gst_return_data.get("input_gst", 0.0),
                "output": gst_return_data.get("output_gst", 0.0),
                "payment": gst_return_data.get("gst_payment", 0.0)
            }
            
            # Calculate differences
            differences = {
                "input": bank_totals["input"] - gst_totals["input"],
                "output": bank_totals["output"] - gst_totals["output"],
                "payment": bank_totals["payment"] - gst_totals["payment"]
            }
            
            return {
                "bank_totals": bank_totals,
                "gst_totals": gst_totals,
                "differences": differences,
                "transactions": [t.to_dict() for t in transactions]
            }
            
        except DatabaseError as e:
            logger.error(f"Error reconciling GST: {e}")
            raise