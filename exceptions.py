"""
Custom exceptions for VaarthaAI.
Centralizes error definitions for better error handling across the application.
"""

class VaarthaError(Exception):
    """Base exception for all VaarthaAI errors"""
    pass

# Parser exceptions
class ParseError(VaarthaError):
    """Raised when parsing a bank statement fails"""
    pass

class UnsupportedBankError(ParseError):
    """Raised when trying to parse a statement from an unsupported bank"""
    pass

class InvalidStatementFormatError(ParseError):
    """Raised when the statement format is invalid or corrupted"""
    pass

# Classification exceptions
class ClassificationError(VaarthaError):
    """Raised when transaction classification fails"""
    pass

class APIError(VaarthaError):
    """Raised when an API call fails (e.g., GROQ API)"""
    pass

class RateLimitError(APIError):
    """Raised when an API rate limit is exceeded"""
    pass

# Database exceptions
class DatabaseError(VaarthaError):
    """Raised when a database operation fails"""
    pass

class ConnectionError(DatabaseError):
    """Raised when a database connection cannot be established"""
    pass

class TransactionNotFoundError(DatabaseError):
    """Raised when a requested transaction is not found"""
    pass

# RAG exceptions
class RAGError(VaarthaError):
    """Base exception for RAG system errors"""
    pass

class VectorDBError(RAGError):
    """Raised when vector database operations fail"""
    pass

class EmbeddingError(RAGError):
    """Raised when generating embeddings fails"""
    pass

# Model exceptions
class ModelError(VaarthaError):
    """Raised when model operations fail"""
    pass

class BERTError(ModelError):
    """Raised when BERT operations fail"""
    pass

class MLModelError(ModelError):
    """Raised when ML model operations fail"""
    pass

# File system exceptions
class FileSystemError(VaarthaError):
    """Raised when file system operations fail"""
    pass

class PermissionError(FileSystemError):
    """Raised when permission is denied for a file operation"""
    pass