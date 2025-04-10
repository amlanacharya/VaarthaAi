"""
Database utility for storing and retrieving transactions.
Implements thread-safe connection management and proper error handling.
"""

import os
import json
import sqlite3
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
from contextlib import contextmanager

from config import config
from exceptions import DatabaseError, ConnectionError, TransactionNotFoundError
from models.transaction import Transaction, TransactionBatch, TransactionType, TransactionCategory

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOGGING_LEVEL))
logger = logging.getLogger(__name__)


class Database:
    """
    Thread-safe database utility for storing and retrieving transactions.
    Ensures connections are properly closed using context managers.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file. If None, uses the DATABASE_URL from config.
        """
        if db_path is None:
            db_url = config.DATABASE_URL
            if db_url.startswith("sqlite:///"):
                db_path = db_url[10:]

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path
        # Initialize the database schema
        self._initialize_db()

    @contextmanager
    def get_connection(self):
        """
        Get a database connection as a context manager.
        This ensures connections are properly closed after use.
        
        Yields:
            sqlite3.Connection: The database connection
            
        Raises:
            ConnectionError: If the connection cannot be established
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise ConnectionError(f"Failed to connect to database: {e}")
        finally:
            conn.close()

    def _initialize_db(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Create transactions table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    description TEXT NOT NULL,
                    amount REAL NOT NULL,
                    type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT,
                    notes TEXT,
                    source TEXT,
                    confidence REAL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                ''')

                # Create batches table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
                ''')

                # Create batch_transactions table for many-to-many relationship
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_transactions (
                    batch_id TEXT,
                    transaction_id TEXT,
                    PRIMARY KEY (batch_id, transaction_id),
                    FOREIGN KEY (batch_id) REFERENCES batches (id),
                    FOREIGN KEY (transaction_id) REFERENCES transactions (id)
                )
                ''')
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")

    def save_transaction(self, transaction: Transaction) -> str:
        """
        Save a transaction to the database.

        Args:
            transaction: The transaction to save.

        Returns:
            The ID of the saved transaction.
            
        Raises:
            DatabaseError: If the operation fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Generate ID if not present
                if transaction.id is None:
                    import uuid
                    transaction.id = str(uuid.uuid4())

                now = datetime.now().isoformat()

                # Convert metadata to JSON
                metadata_json = json.dumps(transaction.metadata)

                # Insert or update transaction
                cursor.execute('''
                INSERT OR REPLACE INTO transactions
                (id, date, description, amount, type, category, subcategory, notes, source, confidence, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transaction.id,
                    transaction.date.isoformat(),
                    transaction.description,
                    transaction.amount,
                    transaction.type.value,
                    transaction.category.value,
                    transaction.subcategory,
                    transaction.notes,
                    transaction.source,
                    transaction.confidence,
                    metadata_json,
                    now,
                    now
                ))

                return transaction.id
        except Exception as e:
            logger.error(f"Failed to save transaction: {e}")
            raise DatabaseError(f"Transaction save failed: {e}")

    def save_batch(self, batch: TransactionBatch) -> str:
        """
        Save a batch of transactions to the database.

        Args:
            batch: The transaction batch to save.

        Returns:
            The ID of the saved batch.
            
        Raises:
            DatabaseError: If the operation fails
        """
        try:
            # Generate batch ID
            import uuid
            batch_id = str(uuid.uuid4())

            # First, save the batch header
            with self.get_connection() as conn:
                cursor = conn.cursor()

                now = datetime.now().isoformat()

                # Convert metadata to JSON
                metadata_json = json.dumps(batch.metadata)

                # Insert batch
                cursor.execute('''
                INSERT INTO batches (id, source, metadata, created_at)
                VALUES (?, ?, ?, ?)
                ''', (
                    batch_id,
                    batch.source,
                    metadata_json,
                    now
                ))

            # Save each transaction separately
            transaction_ids = []
            for transaction in batch.transactions:
                transaction_id = self.save_transaction(transaction)
                transaction_ids.append(transaction_id)

            # Link transactions to batch
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for transaction_id in transaction_ids:
                    cursor.execute('''
                    INSERT INTO batch_transactions (batch_id, transaction_id)
                    VALUES (?, ?)
                    ''', (
                        batch_id,
                        transaction_id
                    ))

            return batch_id
        except Exception as e:
            logger.error(f"Failed to save batch: {e}")
            raise DatabaseError(f"Batch save failed: {e}")

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """
        Retrieve a transaction by ID.

        Args:
            transaction_id: The ID of the transaction to retrieve.

        Returns:
            The transaction if found, None otherwise.
            
        Raises:
            TransactionNotFoundError: If the transaction doesn't exist
            DatabaseError: If the operation fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                SELECT * FROM transactions WHERE id = ?
                ''', (transaction_id,))

                row = cursor.fetchone()
                if row is None:
                    raise TransactionNotFoundError(f"Transaction with ID {transaction_id} not found")

                return self._row_to_transaction(dict(row))
        except TransactionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving transaction: {e}")
            raise DatabaseError(f"Failed to retrieve transaction: {e}")

    def get_transactions(self, limit: int = 100, offset: int = 0, 
                         category: Optional[Union[TransactionCategory, str]] = None,
                         transaction_type: Optional[Union[TransactionType, str]] = None) -> List[Transaction]:
        """
        Retrieve a list of transactions with optional filtering.

        Args:
            limit: Maximum number of transactions to retrieve.
            offset: Number of transactions to skip.
            category: Filter by transaction category.
            transaction_type: Filter by transaction type.

        Returns:
            List of transactions.
            
        Raises:
            DatabaseError: If the operation fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                query = '''
                SELECT * FROM transactions
                '''
                params = []
                
                # Add filters if provided
                where_clauses = []
                if category:
                    category_value = category.value if isinstance(category, TransactionCategory) else category
                    where_clauses.append("category = ?")
                    params.append(category_value)
                    
                if transaction_type:
                    type_value = transaction_type.value if isinstance(transaction_type, TransactionType) else transaction_type
                    where_clauses.append("type = ?")
                    params.append(type_value)
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
                
                query += '''
                ORDER BY date DESC
                LIMIT ? OFFSET ?
                '''
                params.extend([limit, offset])

                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_transaction(dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Error retrieving transactions: {e}")
            raise DatabaseError(f"Failed to retrieve transactions: {e}")

    def get_batch(self, batch_id: str) -> Optional[TransactionBatch]:
        """
        Retrieve a batch by ID.

        Args:
            batch_id: The ID of the batch to retrieve.

        Returns:
            The batch if found, None otherwise.
            
        Raises:
            DatabaseError: If the operation fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Get batch info
                cursor.execute('''
                SELECT * FROM batches WHERE id = ?
                ''', (batch_id,))

                row = cursor.fetchone()
                if row is None:
                    return None

                # Convert row to dictionary
                batch_data = dict(row)

                # Parse metadata
                batch_data['metadata'] = json.loads(batch_data['metadata'])

                # Get transactions in this batch
                cursor.execute('''
                SELECT t.* FROM transactions t
                JOIN batch_transactions bt ON t.id = bt.transaction_id
                WHERE bt.batch_id = ?
                ORDER BY t.date DESC
                ''', (batch_id,))

                transaction_rows = cursor.fetchall()
                transactions = [self._row_to_transaction(dict(row)) for row in transaction_rows]

                return TransactionBatch(
                    transactions=transactions,
                    source=batch_data['source'],
                    metadata=batch_data['metadata']
                )
        except Exception as e:
            logger.error(f"Error retrieving batch: {e}")
            raise DatabaseError(f"Failed to retrieve batch: {e}")

    def get_batches(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve a list of batch summaries.

        Args:
            limit: Maximum number of batches to retrieve.
            offset: Number of batches to skip.

        Returns:
            List of batch summaries.
            
        Raises:
            DatabaseError: If the operation fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                SELECT b.id, b.source, b.created_at, COUNT(bt.transaction_id) as transaction_count
                FROM batches b
                LEFT JOIN batch_transactions bt ON b.id = bt.batch_id
                GROUP BY b.id
                ORDER BY b.created_at DESC
                LIMIT ? OFFSET ?
                ''', (limit, offset))

                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error retrieving batches: {e}")
            raise DatabaseError(f"Failed to retrieve batches: {e}")

    def delete_transaction(self, transaction_id: str) -> bool:
        """
        Delete a transaction by ID.

        Args:
            transaction_id: The ID of the transaction to delete.

        Returns:
            True if the transaction was deleted, False otherwise.
            
        Raises:
            DatabaseError: If the operation fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # First remove from batch_transactions
                cursor.execute('''
                DELETE FROM batch_transactions WHERE transaction_id = ?
                ''', (transaction_id,))
                
                # Then delete the transaction
                cursor.execute('''
                DELETE FROM transactions WHERE id = ?
                ''', (transaction_id,))
                
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting transaction: {e}")
            raise DatabaseError(f"Failed to delete transaction: {e}")

    def delete_batch(self, batch_id: str, delete_transactions: bool = False) -> bool:
        """
        Delete a batch by ID.

        Args:
            batch_id: The ID of the batch to delete.
            delete_transactions: If True, also delete all transactions in the batch.

        Returns:
            True if the batch was deleted, False otherwise.
            
        Raises:
            DatabaseError: If the operation fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if delete_transactions:
                    # Get transaction IDs in this batch
                    cursor.execute('''
                    SELECT transaction_id FROM batch_transactions WHERE batch_id = ?
                    ''', (batch_id,))
                    
                    transaction_ids = [row[0] for row in cursor.fetchall()]
                    
                    # Delete transactions
                    for transaction_id in transaction_ids:
                        cursor.execute('''
                        DELETE FROM transactions WHERE id = ?
                        ''', (transaction_id,))
                
                # Delete batch associations
                cursor.execute('''
                DELETE FROM batch_transactions WHERE batch_id = ?
                ''', (batch_id,))
                
                # Delete the batch
                cursor.execute('''
                DELETE FROM batches WHERE id = ?
                ''', (batch_id,))
                
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting batch: {e}")
            raise DatabaseError(f"Failed to delete batch: {e}")

    def _row_to_transaction(self, row: Dict[str, Any]) -> Transaction:
        """
        Convert a database row to a Transaction object.
        
        Args:
            row: Dictionary containing transaction data from database
            
        Returns:
            Transaction: The transaction object
        """
        # Parse date
        date_str = row.get('date')
        if date_str:
            try:
                date = datetime.fromisoformat(date_str)
            except ValueError:
                logger.warning(f"Invalid date format: {date_str}, using current date")
                date = datetime.now()
        else:
            date = datetime.now()

        # Parse metadata
        metadata_str = row.get('metadata')
        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
        except json.JSONDecodeError:
            logger.warning(f"Invalid metadata JSON: {metadata_str}, using empty dict")
            metadata = {}

        # Create Transaction object
        return Transaction(
            id=row.get('id'),
            date=date,
            description=row.get('description', ''),
            amount=float(row.get('amount', 0.0)),
            type=TransactionType(row.get('type')) if row.get('type') else TransactionType.DEBIT,
            category=TransactionCategory(row.get('category')) if row.get('category') else TransactionCategory.UNCATEGORIZED,
            subcategory=row.get('subcategory'),
            notes=row.get('notes'),
            source=row.get('source'),
            confidence=float(row.get('confidence', 0.0)),
            metadata=metadata
        )