"""
Tests for the transaction natural language query module.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from transaction_nl_query import TransactionNLQuery
from models.transaction import Transaction, TransactionType, TransactionCategory


def test_transaction_nl_query_initialization():
    """Test that the TransactionNLQuery initializes correctly."""
    with patch('transaction_nl_query.Chroma'):
        with patch('transaction_nl_query.HuggingFaceEmbeddings'):
            query_processor = TransactionNLQuery(db_path=":memory:")
            
            assert hasattr(query_processor, 'db_path')
            assert hasattr(query_processor, 'vector_store_path')
            assert hasattr(query_processor, 'embeddings')


def test_is_transaction_query():
    """Test the query classification functionality."""
    with patch('transaction_nl_query.Chroma'):
        with patch('transaction_nl_query.HuggingFaceEmbeddings'):
            query_processor = TransactionNLQuery(db_path=":memory:")
            
            # Transaction-related queries
            assert query_processor.is_transaction_query("What is my highest expense?") == True
            assert query_processor.is_transaction_query("How much did I spend on rent?") == True
            assert query_processor.is_transaction_query("Show me all transactions from last month") == True
            assert query_processor.is_transaction_query("What was my total income this year?") == True
            
            # Regulation-related queries
            assert query_processor.is_transaction_query("What are the GST filing requirements?") == False
            assert query_processor.is_transaction_query("Explain section 80C of the income tax act") == False
            assert query_processor.is_transaction_query("What are the compliance requirements for TDS?") == False


def test_rule_based_sql_generation():
    """Test the rule-based SQL generation fallback."""
    with patch('transaction_nl_query.Chroma'):
        with patch('transaction_nl_query.HuggingFaceEmbeddings'):
            query_processor = TransactionNLQuery(db_path=":memory:")
            
            # Test highest expense query
            sql = query_processor.rule_based_sql_generation("What is my highest expense?")
            assert "SELECT" in sql
            assert "ORDER BY amount DESC" in sql
            assert "type = 'debit'" in sql
            
            # Test total income query
            sql = query_processor.rule_based_sql_generation("What is my total income?")
            assert "SELECT SUM(amount)" in sql
            assert "type = 'credit'" in sql
            
            # Test category-specific query
            sql = query_processor.rule_based_sql_generation("How much did I spend on rent?")
            assert "SUM(amount)" in sql
            assert "category = 'expense_rent'" in sql
            
            # Test monthly query
            sql = query_processor.rule_based_sql_generation("Show me my monthly expenses")
            assert "strftime" in sql
            assert "GROUP BY month" in sql


def test_nl_to_sql():
    """Test the natural language to SQL conversion."""
    with patch('transaction_nl_query.Chroma') as mock_chroma:
        with patch('transaction_nl_query.HuggingFaceEmbeddings'):
            # Mock the vector store similarity search
            mock_vector_store = MagicMock()
            mock_vector_store.similarity_search.return_value = [
                MagicMock(page_content="Query: Show me all transactions from last month; SQL: SELECT * FROM transactions WHERE date >= date('now', 'start of month', '-1 month') AND date < date('now', 'start of month');")
            ]
            mock_chroma.return_value = mock_vector_store
            
            # Mock Groq client
            mock_groq_response = MagicMock()
            mock_groq_response.choices = [MagicMock(message=MagicMock(content="SELECT * FROM transactions WHERE date >= date('now', 'start of month', '-1 month') AND date < date('now', 'start of month')"))]
            
            mock_groq_client = MagicMock()
            mock_groq_client.chat.completions.create.return_value = mock_groq_response
            
            # Create query processor with mocked dependencies
            query_processor = TransactionNLQuery(db_path=":memory:")
            query_processor.groq_client = mock_groq_client
            
            # Test natural language to SQL conversion
            sql = query_processor.nl_to_sql("Show me transactions from last month")
            
            # Check that the correct SQL was generated
            assert "SELECT" in sql
            assert "FROM transactions" in sql
            assert "date" in sql
            
            # Verify that the LLM was called with the right prompt
            mock_groq_client.chat.completions.create.assert_called_once()
            args, kwargs = mock_groq_client.chat.completions.create.call_args
            assert "system" in kwargs["messages"][0]["role"]
            assert "user" in kwargs["messages"][1]["role"]
            assert "Generate SQL for: " in kwargs["messages"][1]["content"]


def test_clean_sql_query():
    """Test SQL query cleaning functionality."""
    with patch('transaction_nl_query.Chroma'):
        with patch('transaction_nl_query.HuggingFaceEmbeddings'):
            query_processor = TransactionNLQuery(db_path=":memory:")
            
            # Test cleaning markdown code blocks
            sql_with_markdown = "```sql\nSELECT * FROM transactions WHERE amount > 1000;\n```"
            cleaned_sql = query_processor.clean_sql_query(sql_with_markdown)
            assert cleaned_sql == "SELECT * FROM transactions WHERE amount > 1000;"
            
            # Test cleaning single backticks
            sql_with_backticks = "`SELECT * FROM transactions`"
            cleaned_sql = query_processor.clean_sql_query(sql_with_backticks)
            assert cleaned_sql == "SELECT * FROM transactions"


@pytest.mark.parametrize(
    "query,expected_contains",
    [
        ("What is my highest expense?", "ORDER BY amount DESC"),
        ("Show my total income", "SUM(amount)"),
        ("List transactions from last month", "date"),
        ("Average expense amount by category", "AVG(amount)"),
    ]
)
def test_process_query(query, expected_contains):
    """Test the end-to-end query processing with parameterized test cases."""
    with patch('transaction_nl_query.Chroma'):
        with patch('transaction_nl_query.HuggingFaceEmbeddings'):
            with patch('transaction_nl_query.TransactionNLQuery.execute_query') as mock_execute:
                # Mock the execute_query method to return a sample DataFrame
                mock_execute.return_value = (
                    pd.DataFrame({
                        "amount": [1000, 2000, 3000],
                        "description": ["Rent", "Utilities", "Salary"],
                        "date": ["2023-01-01", "2023-01-15", "2023-01-31"]
                    }),
                    f"SELECT * FROM transactions WHERE {expected_contains} LIMIT 10"
                )
                
                # Mock the nl_to_sql method to return the expected SQL
                with patch('transaction_nl_query.TransactionNLQuery.nl_to_sql') as mock_nl_to_sql:
                    mock_nl_to_sql.return_value = f"SELECT * FROM transactions WHERE {expected_contains} LIMIT 10"
                    
                    # Create the query processor
                    query_processor = TransactionNLQuery(db_path=":memory:")
                    
                    # Process the query
                    result = query_processor.process_query(query)
                    
                    # Check that we got a result
                    assert "data" in result
                    assert len(result["data"]) == 3
                    assert "sql" in result
                    assert expected_contains in result["sql"]
                    assert "is_transaction_query" in result
                    assert result["is_transaction_query"] == True