import os
import json
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from models.transaction import Transaction, TransactionBatch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    """
    Simple database utility for storing and retrieving transactions.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file. If None, uses the DATABASE_URL from environment.
        """
        if db_path is None:
            db_url = os.getenv("DATABASE_URL", "sqlite:///data/vaartha.db")
            if db_url.startswith("sqlite:///"):
                db_path = db_url[10:]

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path
        # Initialize the database schema
        self.initialize_db()

    def get_connection(self):
        """
        Get a new database connection for thread safety.
        Each operation will use its own connection to avoid thread issues.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_db(self):
        """Initialize the database schema if it doesn't exist."""
        conn = self.get_connection()
        try:
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

            conn.commit()
        finally:
            conn.close()

    def save_transaction(self, transaction: Transaction) -> str:
        """
        Save a transaction to the database.

        Args:
            transaction: The transaction to save.

        Returns:
            The ID of the saved transaction.
        """
        conn = self.get_connection()
        try:
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

            conn.commit()
            return transaction.id
        finally:
            conn.close()

    def save_batch(self, batch: TransactionBatch) -> str:
        """
        Save a batch of transactions to the database.

        Args:
            batch: The transaction batch to save.

        Returns:
            The ID of the saved batch.
        """
        # Generate batch ID
        import uuid
        batch_id = str(uuid.uuid4())

        # First, save the batch header
        conn = self.get_connection()
        try:
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

            conn.commit()
        finally:
            conn.close()

        # Save each transaction separately (each with its own connection)
        for transaction in batch.transactions:
            transaction_id = self.save_transaction(transaction)

            # Link transaction to batch with a new connection
            conn = self.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO batch_transactions (batch_id, transaction_id)
                VALUES (?, ?)
                ''', (
                    batch_id,
                    transaction_id
                ))
                conn.commit()
            finally:
                conn.close()

        return batch_id

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """
        Retrieve a transaction by ID.

        Args:
            transaction_id: The ID of the transaction to retrieve.

        Returns:
            The transaction if found, None otherwise.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            SELECT * FROM transactions WHERE id = ?
            ''', (transaction_id,))

            row = cursor.fetchone()
            if row is None:
                return None

            # Convert row to dictionary
            data = dict(row)

            # Parse date
            data['date'] = datetime.fromisoformat(data['date'])

            # Parse metadata
            data['metadata'] = json.loads(data['metadata'])

            # Remove database-specific fields
            data.pop('created_at', None)
            data.pop('updated_at', None)

            return Transaction.from_dict(data)
        finally:
            conn.close()

    def get_transactions(self, limit: int = 100, offset: int = 0) -> List[Transaction]:
        """
        Retrieve a list of transactions.

        Args:
            limit: Maximum number of transactions to retrieve.
            offset: Number of transactions to skip.

        Returns:
            List of transactions.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            SELECT * FROM transactions
            ORDER BY date DESC
            LIMIT ? OFFSET ?
            ''', (limit, offset))

            rows = cursor.fetchall()
            transactions = []

            for row in rows:
                # Convert row to dictionary
                data = dict(row)

                # Parse date
                data['date'] = datetime.fromisoformat(data['date'])

                # Parse metadata
                data['metadata'] = json.loads(data['metadata'])

                # Remove database-specific fields
                data.pop('created_at', None)
                data.pop('updated_at', None)

                transactions.append(Transaction.from_dict(data))

            return transactions
        finally:
            conn.close()

    def get_batch(self, batch_id: str) -> Optional[TransactionBatch]:
        """
        Retrieve a batch by ID.

        Args:
            batch_id: The ID of the batch to retrieve.

        Returns:
            The batch if found, None otherwise.
        """
        conn = self.get_connection()
        try:
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
            transactions = []

            for row in transaction_rows:
                # Convert row to dictionary
                data = dict(row)

                # Parse date
                data['date'] = datetime.fromisoformat(data['date'])

                # Parse metadata
                data['metadata'] = json.loads(data['metadata'])

                # Remove database-specific fields
                data.pop('created_at', None)
                data.pop('updated_at', None)

                transactions.append(Transaction.from_dict(data))

            return TransactionBatch(
                transactions=transactions,
                source=batch_data['source'],
                metadata=batch_data['metadata']
            )
        finally:
            conn.close()

    def get_batches(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve a list of batch summaries.

        Args:
            limit: Maximum number of batches to retrieve.
            offset: Number of batches to skip.

        Returns:
            List of batch summaries.
        """
        conn = self.get_connection()
        try:
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
            batches = []

            for row in rows:
                batches.append(dict(row))

            return batches
        finally:
            conn.close()

    def close(self):
        """Close the database connection (legacy method, kept for compatibility)."""
        # This method is no longer needed since we create a new connection for each operation
        # but we keep it for backward compatibility
        pass
