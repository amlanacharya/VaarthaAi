import pandas as pd
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from models.transaction import Transaction, TransactionType, TransactionBatch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BankStatementParser:
    """
    Parser for bank statements from various Indian banks.
    """

    def __init__(self):
        """Initialize the bank statement parser."""
        # Map of bank names to their parsing functions
        self.parsers = {
            "hdfc": self.parse_hdfc,
            "sbi": self.parse_sbi,
            "icici": self.parse_icici,
            "axis": self.parse_axis,
            # Add more banks as needed
        }

    def detect_bank(self, file_path: str) -> Optional[str]:
        """
        Detect which bank the statement is from based on content patterns.

        Args:
            file_path: Path to the bank statement file.

        Returns:
            The detected bank name or None if not detected.
        """
        try:
            # Read the first few lines of the file to detect patterns
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = ''.join(f.readline() for _ in range(10)).lower()

            # Check for bank-specific patterns
            if 'hdfc bank' in header:
                return "hdfc"
            elif 'state bank of india' in header or 'sbi' in header:
                return "sbi"
            elif 'icici bank' in header:
                return "icici"
            elif 'axis bank' in header:
                return "axis"

            return None

        except Exception as e:
            logger.error(f"Error detecting bank from statement: {e}")
            return None

    def parse(self, file_path: str, bank_name: Optional[str] = None) -> TransactionBatch:
        """
        Parse a bank statement file into a batch of transactions.

        Args:
            file_path: Path to the bank statement file.
            bank_name: Name of the bank (if known). If None, will attempt to detect.

        Returns:
            A TransactionBatch containing the parsed transactions.
        """
        # Detect bank if not provided
        if not bank_name:
            bank_name = self.detect_bank(file_path)
            if not bank_name:
                raise ValueError("Could not detect bank type. Please specify bank_name.")

        # Get the appropriate parser function
        parser_func = self.parsers.get(bank_name.lower())
        if not parser_func:
            raise ValueError(f"No parser available for bank: {bank_name}")

        # Parse the file
        transactions = parser_func(file_path)

        # Create and return a transaction batch
        return TransactionBatch(
            transactions=transactions,
            source=f"{bank_name.upper()} Bank Statement",
            metadata={"file_path": file_path, "parsed_at": datetime.now().isoformat()}
        )

    def parse_hdfc(self, file_path: str) -> List[Transaction]:
        """
        Parse HDFC Bank statement.

        Args:
            file_path: Path to the HDFC bank statement file.

        Returns:
            List of parsed transactions.
        """
        try:
            # For CSV format
            if file_path.endswith('.csv'):
                # Read the file content to determine format
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Determine how many header rows to skip
                if 'HDFC BANK STATEMENT' in content:
                    # Count header lines before the column headers
                    header_lines = 0
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            header_lines += 1
                            if 'date,narration' in line.lower():
                                break

                    # Read the CSV skipping the header lines
                    df = pd.read_csv(file_path, skiprows=range(header_lines-1), encoding='utf-8')
                else:
                    # Standard HDFC format
                    df = pd.read_csv(file_path, skiprows=range(15), encoding='utf-8')

                # Clean column names
                df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

                # Print column names for debugging
                print(f"Columns found: {df.columns.tolist()}")

                # Expected columns: date, narration, chq/ref_no, value_date, withdrawal_amt, deposit_amt, closing_balance
                transactions = []

                # Skip the first row if it's a balance brought forward
                start_idx = 0
                if len(df) > 0 and 'BALANCE' in str(df.iloc[0].get('narration', '')).upper():
                    start_idx = 1

                for idx in range(start_idx, len(df)):
                    row = df.iloc[idx]

                    # Skip rows with NaN or empty values
                    if pd.isna(row.get('date', '')) or str(row.get('date', '')).strip() == '':
                        continue

                    # Determine transaction type
                    withdrawal = row.get('withdrawal_amt', 0)
                    deposit = row.get('deposit_amt', 0)

                    # Convert to float if string
                    if isinstance(withdrawal, str):
                        withdrawal = float(withdrawal.replace(',', '') or 0)
                    if isinstance(deposit, str):
                        deposit = float(deposit.replace(',', '') or 0)

                    if pd.notna(withdrawal) and withdrawal > 0:
                        amount = float(withdrawal)
                        type_val = TransactionType.DEBIT
                    elif pd.notna(deposit) and deposit > 0:
                        amount = float(deposit)
                        type_val = TransactionType.CREDIT
                    else:
                        continue  # Skip rows without amount

                    # Parse date
                    date_str = str(row.get('date', ''))
                    try:
                        date = datetime.strptime(date_str, '%d/%m/%y')
                    except ValueError:
                        try:
                            date = datetime.strptime(date_str, '%d/%m/%Y')
                        except ValueError:
                            print(f"Could not parse date: {date_str}")
                            date = datetime.now()  # Fallback

                    # Get description
                    description = str(row.get('narration', ''))

                    # Create transaction
                    transaction = Transaction(
                        id=str(uuid.uuid4()),
                        date=date,
                        description=description,
                        amount=amount,
                        type=type_val,
                        source="HDFC Bank Statement",
                        metadata={
                            "reference_no": str(row.get('chq/ref_no', '')),
                            "value_date": str(row.get('value_date', '')),
                            "closing_balance": float(row.get('closing_balance', 0))
                        }
                    )

                    transactions.append(transaction)

                print(f"Parsed {len(transactions)} transactions from HDFC statement")
                return transactions

            # For PDF format (simplified - in a real implementation, would use a PDF parsing library)
            else:
                logger.warning("PDF parsing for HDFC statements is not implemented. Please convert to CSV.")
                return []

        except Exception as e:
            logger.error(f"Error parsing HDFC statement: {e}")
            import traceback
            traceback.print_exc()
            return []

    def parse_sbi(self, file_path: str) -> List[Transaction]:
        """
        Parse SBI Bank statement.

        Args:
            file_path: Path to the SBI bank statement file.

        Returns:
            List of parsed transactions.
        """
        # Simplified implementation - would be expanded for actual use
        logger.info("SBI statement parsing not fully implemented.")
        return []

    def parse_icici(self, file_path: str) -> List[Transaction]:
        """
        Parse ICICI Bank statement.

        Args:
            file_path: Path to the ICICI bank statement file.

        Returns:
            List of parsed transactions.
        """
        # Simplified implementation - would be expanded for actual use
        logger.info("ICICI statement parsing not fully implemented.")
        return []

    def parse_axis(self, file_path: str) -> List[Transaction]:
        """
        Parse Axis Bank statement.

        Args:
            file_path: Path to the Axis bank statement file.

        Returns:
            List of parsed transactions.
        """
        # Simplified implementation - would be expanded for actual use
        logger.info("Axis statement parsing not fully implemented.")
        return []


# Function to generate sample transactions for testing
def generate_sample_transactions(count: int = 20) -> List[Transaction]:
    """
    Generate sample transactions for testing.

    Args:
        count: Number of transactions to generate.

    Returns:
        List of sample transactions.
    """
    import random

    descriptions = [
        "SALARY CREDIT",
        "RENT PAYMENT",
        "ELECTRICITY BILL PAYMENT",
        "OFFICE SUPPLIES PURCHASE",
        "GST PAYMENT",
        "INCOME TAX PAYMENT",
        "CONSULTANT FEE",
        "INTERNET BILL",
        "MOBILE RECHARGE",
        "TRAVEL EXPENSES",
        "RESTAURANT BILL",
        "FUEL EXPENSE",
        "INSURANCE PREMIUM",
        "SOFTWARE SUBSCRIPTION",
        "MAINTENANCE CHARGES",
        "ADVERTISING EXPENSE",
        "CLIENT PAYMENT RECEIVED",
        "INTEREST CREDIT",
        "DIVIDEND INCOME",
        "TRANSFER TO SAVINGS"
    ]

    transactions = []

    for i in range(count):
        # Random date in the last 3 months
        days_ago = random.randint(1, 90)
        date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        date = date.replace(day=date.day - days_ago)

        # Random amount between 1000 and 50000
        amount = round(random.uniform(1000, 50000), 2)

        # Random transaction type
        type_val = random.choice(list(TransactionType))

        # Random description
        description = random.choice(descriptions)

        transaction = Transaction(
            id=str(uuid.uuid4()),
            date=date,
            description=description,
            amount=amount,
            type=type_val,
            source="Sample Data"
        )

        transactions.append(transaction)

    return transactions
