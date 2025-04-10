from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class TransactionType(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"


class TransactionCategory(str, Enum):
    # GST Categories
    GST_INPUT = "gst_input"
    GST_OUTPUT = "gst_output"
    
    # Income Categories
    INCOME_BUSINESS = "income_business"
    INCOME_INTEREST = "income_interest"
    INCOME_RENT = "income_rent"
    INCOME_OTHER = "income_other"
    
    # Expense Categories
    EXPENSE_RENT = "expense_rent"
    EXPENSE_SALARY = "expense_salary"
    EXPENSE_UTILITIES = "expense_utilities"
    EXPENSE_OFFICE_SUPPLIES = "expense_office_supplies"
    EXPENSE_TRAVEL = "expense_travel"
    EXPENSE_MEALS = "expense_meals"
    EXPENSE_ADVERTISING = "expense_advertising"
    EXPENSE_PROFESSIONAL_SERVICES = "expense_professional_services"
    EXPENSE_INSURANCE = "expense_insurance"
    EXPENSE_MAINTENANCE = "expense_maintenance"
    EXPENSE_OTHER = "expense_other"
    
    # Tax Categories
    TAX_INCOME = "tax_income"
    TAX_GST = "tax_gst"
    TAX_TDS = "tax_tds"
    TAX_OTHER = "tax_other"
    
    # Personal Categories
    PERSONAL = "personal"
    
    # Other
    TRANSFER = "transfer"
    UNCATEGORIZED = "uncategorized"


@dataclass
class Transaction:
    """Represents a financial transaction."""
    
    id: Optional[str] = None
    date: datetime = field(default_factory=datetime.now)
    description: str = ""
    amount: float = 0.0
    type: TransactionType = TransactionType.DEBIT
    category: TransactionCategory = TransactionCategory.UNCATEGORIZED
    subcategory: Optional[str] = None
    notes: Optional[str] = None
    source: Optional[str] = None  # e.g., "HDFC Bank Statement"
    confidence: float = 0.0  # Classification confidence score
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary."""
        return {
            "id": self.id,
            "date": self.date.isoformat() if self.date else None,
            "description": self.description,
            "amount": self.amount,
            "type": self.type.value if self.type else None,
            "category": self.category.value if self.category else None,
            "subcategory": self.subcategory,
            "notes": self.notes,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create transaction from dictionary."""
        if "date" in data and isinstance(data["date"], str):
            data["date"] = datetime.fromisoformat(data["date"])
        
        if "type" in data and isinstance(data["type"], str):
            data["type"] = TransactionType(data["type"])
            
        if "category" in data and isinstance(data["category"], str):
            data["category"] = TransactionCategory(data["category"])
            
        return cls(**data)


@dataclass
class TransactionBatch:
    """A collection of transactions, typically from a single source."""
    
    transactions: List[Transaction] = field(default_factory=list)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_transaction(self, transaction: Transaction) -> None:
        """Add a transaction to the batch."""
        self.transactions.append(transaction)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary."""
        return {
            "transactions": [t.to_dict() for t in self.transactions],
            "source": self.source,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionBatch':
        """Create batch from dictionary."""
        transactions = [Transaction.from_dict(t) for t in data.get("transactions", [])]
        return cls(
            transactions=transactions,
            source=data.get("source"),
            metadata=data.get("metadata", {})
        )
