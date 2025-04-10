"""
Natural language query processor for VaarthaAI transaction data.
Allows users to ask questions about their transaction data in plain language.
"""

import os
import re
import json
import logging
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta

from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from models.transaction import TransactionType, TransactionCategory
from config import config

# Configure logging
logger = logging.getLogger(__name__)

class TransactionNLQuery:
    """
    Process natural language queries about transaction data.
    Uses a combination of RAG and LLM to generate SQL queries from natural language.
    """
    
    def __init__(self, db_path: Optional[str] = None, vector_store_path: str = "data/transaction_query_vectorstore"):
        """
        Initialize the transaction NL query processor.
        
        Args:
            db_path: Path to the SQLite database containing transaction data
            vector_store_path: Path to store/load the vector store for query examples
        """
        # Set up database path
        if db_path is None:
            db_url = config.DATABASE_URL
            if db_url.startswith("sqlite:///"):
                db_path = db_url[10:]
        
        self.db_path = db_path
        self.vector_store_path = vector_store_path
        
        # Initialize Groq client if API key is available
        self.groq_client = None
        groq_api_key = os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
            logger.info("Initialized Groq client for NL queries")
        else:
            logger.warning("No Groq API key found. Using rule-based SQL generation only.")
        
        # Initialize sentence transformer embeddings
        model_name = "all-MiniLM-L6-v2"  # Smaller, efficient model
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize database schema cache
        self._db_schema = None
        
        # Initialize or load vector store
        self.vector_store = self.load_or_create_vector_store()
    
    def load_or_create_vector_store(self) -> Chroma:
        """Load existing vector store or create a new one."""
        if os.path.exists(self.vector_store_path) and os.listdir(self.vector_store_path):
            logger.info("Loading existing transaction query vector store...")
            return Chroma(persist_directory=self.vector_store_path, embedding_function=self.embeddings)
        else:
            logger.info("Creating new transaction query vector store...")
            return self.create_vector_store()
    
    def create_vector_store(self) -> Chroma:
        """Create vector store from database schema and example queries."""
        # Get schema information
        schema = self.get_db_schema()
        
        # Generate example queries
        example_queries = self.create_example_queries()
        
        # Combine all documents
        documents = [schema] + example_queries
        
        # Add financial-specific examples for transactions
        financial_examples = [
            "Query: Show all expenses greater than ₹2000; SQL: SELECT * FROM transactions WHERE type = 'debit' AND amount > 2000;",
            "Query: Calculate total income by month; SQL: SELECT strftime('%Y-%m', date) as month, SUM(amount) as total FROM transactions WHERE type = 'credit' GROUP BY month ORDER BY month;",
            "Query: Find average expense amount by category; SQL: SELECT category, AVG(amount) as avg_amount FROM transactions WHERE type = 'debit' GROUP BY category;",
            "Query: List all expenses in the Office Supplies category; SQL: SELECT * FROM transactions WHERE category = 'expense_office_supplies';",
            "Query: Count transactions by category; SQL: SELECT category, COUNT(*) as count FROM transactions GROUP BY category;",
            "Query: Show monthly expense totals for the second quarter; SQL: SELECT strftime('%Y-%m', date) as month, SUM(amount) as total FROM transactions WHERE type = 'debit' AND date BETWEEN '2023-04-01' AND '2023-06-30' GROUP BY month;"
        ]
        
        documents.extend(financial_examples)
        
        # Use Indian financial examples
        indian_financial_examples = [
            "Query: Show all GST related transactions; SQL: SELECT * FROM transactions WHERE category LIKE '%gst%';",
            "Query: Calculate total TDS payments; SQL: SELECT SUM(amount) as total_tds FROM transactions WHERE category = 'tax_tds';",
            "Query: Show rent expenses for the last quarter; SQL: SELECT * FROM transactions WHERE category = 'expense_rent' AND date >= date('now', '-3 months');",
            "Query: What is my total income this year; SQL: SELECT SUM(amount) as total_income FROM transactions WHERE type = 'credit' AND date >= date('now', 'start of year');",
            "Query: List my highest expenses; SQL: SELECT * FROM transactions WHERE type = 'debit' ORDER BY amount DESC LIMIT 5;"
        ]
        
        documents.extend(indian_financial_examples)
        
        # Create vector store
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = []
        for doc in documents:
            texts.extend(text_splitter.split_text(doc))
        
        # Save to disk for reuse
        vector_store = Chroma.from_texts(texts, self.embeddings, persist_directory=self.vector_store_path)
        vector_store.persist()
        
        return vector_store
    
    def get_db_schema(self) -> str:
        """Extract schema from transaction database."""
        if self._db_schema is not None:
            return self._db_schema
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get transaction table info
            cursor.execute("PRAGMA table_info(transactions);")
            columns = cursor.fetchall()
            col_info = [f"{col[1]} ({col[2]})" for col in columns]
            
            schema = ["Database Schema for Financial Transactions:"]
            schema.append("Table: transactions")
            schema.append(f"Columns: {', '.join(col_info)}")
            
            # Add information about transaction types
            schema.append("\nTransaction Types:")
            for t_type in TransactionType:
                schema.append(f"- {t_type.value}: {'Income' if t_type == TransactionType.CREDIT else 'Expense'}")
            
            # Add information about transaction categories
            schema.append("\nTransaction Categories:")
            for category in TransactionCategory:
                schema.append(f"- {category.value}: {category.name}")
            
            # Add sample data
            cursor.execute("SELECT * FROM transactions LIMIT 3")
            sample_data = cursor.fetchall()
            
            if sample_data:
                column_names = [col[1] for col in columns]
                sample_rows = []
                for row in sample_data:
                    sample_row = {column_names[i]: row[i] for i in range(len(column_names))}
                    sample_rows.append(sample_row)
                schema.append(f"\nSample data: {json.dumps(sample_rows, default=str)}")
            
            conn.close()
            
            self._db_schema = "\n".join(schema)
            return self._db_schema
            
        except Exception as e:
            logger.error(f"Error getting database schema: {e}")
            return "Error: Could not retrieve database schema."
    
    def create_example_queries(self) -> List[str]:
        """Generate example SQL queries based on transaction data schema."""
        example_queries = [
            "Query: Show me all transactions from last month; SQL: SELECT * FROM transactions WHERE date >= date('now', 'start of month', '-1 month') AND date < date('now', 'start of month');",
            
            "Query: What are my biggest expenses; SQL: SELECT * FROM transactions WHERE type = 'debit' ORDER BY amount DESC LIMIT 10;",
            
            "Query: How much did I spend on rent; SQL: SELECT SUM(amount) as total_rent FROM transactions WHERE category = 'expense_rent';",
            
            "Query: Show me my income sources; SQL: SELECT category, SUM(amount) as total FROM transactions WHERE type = 'credit' GROUP BY category ORDER BY total DESC;",
            
            "Query: What is my monthly spending trend; SQL: SELECT strftime('%Y-%m', date) as month, SUM(amount) as total FROM transactions WHERE type = 'debit' GROUP BY month ORDER BY month;",
            
            "Query: Show transactions with high confidence classification; SQL: SELECT * FROM transactions WHERE confidence > 90;",
            
            "Query: Find uncategorized transactions; SQL: SELECT * FROM transactions WHERE category = 'uncategorized';"
        ]
        
        return example_queries
    
    def is_transaction_query(self, query: str) -> bool:
        """
        Determine if a query is about transaction data rather than regulations.
        
        Args:
            query: The natural language query
            
        Returns:
            True if query is about transaction data, False if about regulations
        """
        # Keywords suggesting transaction data queries
        transaction_keywords = [
            "transaction", "expense", "spent", "income", "received", 
            "payment", "paid", "amount", "money", "rupee", "₹",
            "how much", "total", "average", "highest", "lowest",
            "spend", "earn", "salary", "revenue", "rent", "bill"
        ]
        
        # Check for transaction keywords
        query_lower = query.lower()
        for keyword in transaction_keywords:
            if keyword in query_lower:
                return True
        
        # If no clear transaction keywords, check for regulation keywords
        regulation_keywords = [
            "regulation", "law", "compliance", "tax code", "gst rule", 
            "income tax act", "policy", "requirement", "legal", "statute",
            "provision", "section"
        ]
        
        for keyword in regulation_keywords:
            if keyword in query_lower:
                return False
        
        # If still unclear, default to transaction query for this module
        # The main RAG system will handle regulations queries
        return True
    
    def rule_based_sql_generation(self, query: str) -> str:
        """
        Simple rule-based SQL generation as fallback when no LLM is available.
        
        Args:
            query: The natural language query
            
        Returns:
            Generated SQL query
        """
        query_lower = query.lower()
        
        # Check for common patterns
        if any(term in query_lower for term in ["highest", "largest", "biggest", "most expensive"]):
            if "expense" in query_lower or "spent" in query_lower:
                return "SELECT * FROM transactions WHERE type = 'debit' ORDER BY amount DESC LIMIT 5;"
            elif "income" in query_lower or "received" in query_lower:
                return "SELECT * FROM transactions WHERE type = 'credit' ORDER BY amount DESC LIMIT 5;"
            else:
                return "SELECT * FROM transactions ORDER BY amount DESC LIMIT 5;"
        
        if any(term in query_lower for term in ["total", "sum", "overall"]):
            if "expense" in query_lower or "spent" in query_lower:
                return "SELECT SUM(amount) as total FROM transactions WHERE type = 'debit';"
            elif "income" in query_lower or "received" in query_lower:
                return "SELECT SUM(amount) as total FROM transactions WHERE type = 'credit';"
            elif any(cat.value in query_lower for cat in TransactionCategory):
                for cat in TransactionCategory:
                    if cat.value in query_lower:
                        return f"SELECT SUM(amount) as total FROM transactions WHERE category = '{cat.value}';"
        
        if "average" in query_lower or "avg" in query_lower:
            if "expense" in query_lower or "spent" in query_lower:
                return "SELECT AVG(amount) as average FROM transactions WHERE type = 'debit';"
            elif "income" in query_lower or "received" in query_lower:
                return "SELECT AVG(amount) as average FROM transactions WHERE type = 'credit';"
        
        if "month" in query_lower:
            if "expense" in query_lower or "spent" in query_lower:
                return "SELECT strftime('%Y-%m', date) as month, SUM(amount) as total FROM transactions WHERE type = 'debit' GROUP BY month ORDER BY month;"
            elif "income" in query_lower or "received" in query_lower:
                return "SELECT strftime('%Y-%m', date) as month, SUM(amount) as total FROM transactions WHERE type = 'credit' GROUP BY month ORDER BY month;"
            else:
                return "SELECT strftime('%Y-%m', date) as month, SUM(CASE WHEN type = 'credit' THEN amount ELSE 0 END) as income, SUM(CASE WHEN type = 'debit' THEN amount ELSE 0 END) as expenses FROM transactions GROUP BY month ORDER BY month;"
        
        # Default query
        return "SELECT * FROM transactions ORDER BY date DESC LIMIT 20;"
    
    def nl_to_sql(self, query: str) -> str:
        """
        Convert natural language to SQL using RAG and LLM.
        
        Args:
            query: The natural language query
            
        Returns:
            Generated SQL query
        """
        # Retrieve relevant context from vector store
        relevant_docs = self.vector_store.similarity_search(query, k=5)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Get database schema
        db_schema = self.get_db_schema()
        
        system_prompt = f"""
        You are an expert SQL analyst for financial transaction data. Convert the natural language query to a valid SQLite SQL query.
        
        Database schema:
        {db_schema}
        
        Relevant examples and context:
        {context}
        
        IMPORTANT: 
        - The transactions table contains financial data with columns like id, date, description, amount, type (credit/debit), category, etc.
        - Type 'credit' means income and 'debit' means expense.
        - Categories follow a pattern like 'expense_rent', 'income_business', etc.
        - For date-based queries, use SQLite date functions like strftime().
        - Always check type and category for income/expense related queries.
        
        Return ONLY the SQL query without explanation, comments or markdown formatting.
        """
        
        # Use Groq if available, otherwise use rule-based approach
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generate SQL for: {query}"}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                sql_query = response.choices[0].message.content.strip()
                return self.clean_sql_query(sql_query)
            except Exception as e:
                logger.warning(f"Error using Groq: {e}")
                logger.info("Falling back to rule-based SQL generation...")
                return self.rule_based_sql_generation(query)
        else:
            logger.info("No LLM service available. Using rule-based SQL generation.")
            return self.rule_based_sql_generation(query)
    
    def clean_sql_query(self, sql_query: str) -> str:
        """
        Clean up SQL query by removing markdown formatting and other artifacts.
        
        Args:
            sql_query: Raw SQL query possibly containing markdown
            
        Returns:
            Cleaned SQL query
        """
        # Remove Markdown code block markers with language
        sql_query = re.sub(r'```sql\s*|\s*```', '', sql_query)
        
        # Remove Markdown code block markers without language
        sql_query = re.sub(r'```\s*|\s*```', '', sql_query)
        
        # Remove single backticks
        sql_query = re.sub(r'^`|`$', '', sql_query)
        
        return sql_query.strip()
    
    def execute_query(self, sql_query: str) -> Tuple[pd.DataFrame, str]:
        """
        Execute an SQL query against the transaction database.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Tuple of (DataFrame with results, SQL query used)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Execute the query and convert to DataFrame
            result = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            return result, sql_query
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            raise ValueError(f"Error executing SQL query: {str(e)}")
    
    def format_results(self, result: pd.DataFrame, nl_query: str, sql_query: str) -> Dict[str, Any]:
        """
        Format query results with natural language explanation.
        
        Args:
            result: DataFrame containing query results
            nl_query: Original natural language query
            sql_query: SQL query that was executed
            
        Returns:
            Dictionary with formatted results and metadata
        """
        # Format results for different query types
        formatted_response = {}
        
        # Add raw results
        formatted_response["data"] = result.to_dict(orient="records")
        formatted_response["columns"] = result.columns.tolist()
        formatted_response["row_count"] = len(result)
        formatted_response["sql"] = sql_query
        
        # Generate natural language explanation if Groq is available
        if self.groq_client and not result.empty:
            try:
                # Limit result size for the prompt
                result_sample = result.head(5).to_string()
                if len(result) > 5:
                    result_sample += f"\n\n... and {len(result) - 5} more rows"
                
                system_prompt = """
                You are a financial assistant explaining data query results to a user.
                Provide a short, clear explanation of what the data shows based on their query.
                Focus on insights and key numbers that answer their question.
                Keep your explanation to 2-3 sentences maximum.
                """
                
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"My question was: '{nl_query}'\n\nThe query returned these results:\n{result_sample}\n\nPlease explain what this data shows in 2-3 sentences:"}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                
                explanation = response.choices[0].message.content.strip()
                formatted_response["explanation"] = explanation
            except Exception as e:
                logger.warning(f"Error generating explanation: {e}")
                # Create basic explanation 
                if "sum" in sql_query.lower() or "avg" in sql_query.lower():
                    # For aggregate queries
                    if not result.empty and len(result.columns) > 0:
                        col = result.columns[0]
                        val = result.iloc[0, 0]
                        formatted_response["explanation"] = f"The query returned {col}: {val}"
                elif len(result) == 0:
                    formatted_response["explanation"] = "No matching transactions were found."
                else:
                    formatted_response["explanation"] = f"Found {len(result)} matching transactions."
        else:
            # Create basic explanation without LLM
            if "sum" in sql_query.lower() or "avg" in sql_query.lower():
                # For aggregate queries
                if not result.empty and len(result.columns) > 0:
                    col = result.columns[0]
                    val = result.iloc[0, 0]
                    formatted_response["explanation"] = f"The query returned {col}: {val}"
            elif len(result) == 0:
                formatted_response["explanation"] = "No matching transactions were found."
            else:
                formatted_response["explanation"] = f"Found {len(result)} matching transactions."
        
        return formatted_response
    
    def process_query(self, nl_query: str) -> Dict[str, Any]:
        """
        Process a natural language query from start to finish.
        
        Args:
            nl_query: Natural language query from user
            
        Returns:
            Dictionary with query results and metadata
        """
        try:
            # Check if this is a transaction query
            if not self.is_transaction_query(nl_query):
                return {
                    "error": "This query appears to be about regulations rather than transaction data.",
                    "is_transaction_query": False
                }
            
            # Convert natural language to SQL
            sql_query = self.nl_to_sql(nl_query)
            
            # Execute the query
            result_df, final_sql = self.execute_query(sql_query)
            
            # Format the results
            formatted_results = self.format_results(result_df, nl_query, final_sql)
            formatted_results["is_transaction_query"] = True
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": f"Error processing query: {str(e)}",
                "is_transaction_query": True
            }