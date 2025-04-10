"""
Unified query handler for VaarthaAI.
Combines regulation RAG queries with transaction data natural language queries.
"""

import logging
from typing import Dict, Any, Optional
import os
import sys

# Ensure that the parent directory is in the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from models.rag import FinancialRAG
from transaction_nl_query import TransactionNLQuery
from utils.database import Database

# Configure logging
logger = logging.getLogger(__name__)

class UnifiedQueryHandler:
    """
    Unified query handler that routes questions to the appropriate system:
    - Transaction data questions go to TransactionNLQuery
    - Regulation questions go to FinancialRAG
    """
    
    def __init__(self, db: Optional[Database] = None):
        """
        Initialize the unified query handler.
        
        Args:
            db: Optional database instance for transaction data
        """
        self.transaction_query = TransactionNLQuery()
        self.rag = FinancialRAG()
        self.db = db
        
        # Ensure RAG system is initialized
        if not hasattr(self.rag, 'vectordb') or self.rag.vectordb is None:
            logger.info("Initializing RAG knowledge base")
            self.rag.initialize_knowledge_base()
    
    def query(self, question: str, regulation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user question and route to the appropriate subsystem.
        
        Args:
            question: User's natural language question
            regulation_type: Optional filter for regulation type
            
        Returns:
            Dictionary with query results and metadata
        """
        # First, check if this is a transaction data query
        is_transaction_query = self.transaction_query.is_transaction_query(question)
        
        if is_transaction_query:
            logger.info(f"Processing as transaction query: {question}")
            try:
                # Process with TransactionNLQuery
                result = self.transaction_query.process_query(question)
                
                # If transaction query processor couldn't handle it, fall back to RAG
                if "error" in result and not result.get("is_transaction_query", True):
                    logger.info(f"Falling back to RAG for: {question}")
                    return self.rag.query(question, regulation_type)
                
                # Format response in a way that's compatible with the RAG interface
                formatted_response = {
                    "response": result.get("explanation", ""),
                    "data": result.get("data", []),
                    "columns": result.get("columns", []),
                    "sql": result.get("sql", ""),
                    "is_transaction_data": True
                }
                
                # Add error if present
                if "error" in result:
                    formatted_response["error"] = result["error"]
                
                return formatted_response
                
            except Exception as e:
                logger.error(f"Error in transaction query processing: {e}")
                # Fall back to RAG if transaction query processing fails
                return self.rag.query(question, regulation_type)
        else:
            logger.info(f"Processing as regulation query: {question}")
            # Process with FinancialRAG
            return self.rag.query(question, regulation_type)