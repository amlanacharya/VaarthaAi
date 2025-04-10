import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from groq import Groq
import numpy as np

from models.transaction import Transaction, TransactionCategory, TransactionType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionClassifier:
    """
    Classifies financial transactions using a combination of rule-based and AI approaches.
    """

    def __init__(self, industry: str = "general"):
        """
        Initialize the transaction classifier.

        Args:
            industry: The industry context for classification.
        """
        self.industry = industry
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # Initialize GROQ client if API key is available
        if self.groq_api_key:
            try:
                self.client = Groq(api_key=self.groq_api_key)
                self.has_llm = True
            except Exception as e:
                logger.warning(f"Could not initialize GROQ client: {e}")
                self.has_llm = False
        else:
            logger.warning("No GROQ API key found. Using rule-based classification only.")
            self.has_llm = False

        self.confidence_threshold = 0.85
        self.load_rules()

    def load_rules(self) -> None:
        """Load rule-based classification patterns."""
        # In a real implementation, these would be loaded from a database or file
        self.rules = {
            # GST related keywords
            r"(?i)gst|tax invoice|cgst|sgst|igst": TransactionCategory.GST_INPUT,

            # Income related keywords
            r"(?i)payment received|invoice payment|client payment": TransactionCategory.INCOME_BUSINESS,
            r"(?i)interest|fd interest|savings interest": TransactionCategory.INCOME_INTEREST,
            r"(?i)rent received|rental income": TransactionCategory.INCOME_RENT,

            # Expense related keywords
            r"(?i)rent paid|rent payment": TransactionCategory.EXPENSE_RENT,
            r"(?i)salary|payroll|wages": TransactionCategory.EXPENSE_SALARY,
            r"(?i)electricity|water|gas|internet|phone|utility": TransactionCategory.EXPENSE_UTILITIES,
            r"(?i)office supplies|stationery|printer|toner": TransactionCategory.EXPENSE_OFFICE_SUPPLIES,
            r"(?i)travel|flight|train|taxi|uber|ola": TransactionCategory.EXPENSE_TRAVEL,
            r"(?i)restaurant|food|meal|lunch|dinner|swiggy|zomato": TransactionCategory.EXPENSE_MEALS,
            r"(?i)advertising|marketing|promotion|ad spend": TransactionCategory.EXPENSE_ADVERTISING,
            r"(?i)accountant|lawyer|consultant|professional fee": TransactionCategory.EXPENSE_PROFESSIONAL_SERVICES,
            r"(?i)insurance|policy premium": TransactionCategory.EXPENSE_INSURANCE,
            r"(?i)repair|maintenance|service": TransactionCategory.EXPENSE_MAINTENANCE,

            # Tax related keywords
            r"(?i)income tax|tds|tcs": TransactionCategory.TAX_INCOME,
            r"(?i)gst payment|gst paid": TransactionCategory.TAX_GST,

            # Transfer related keywords
            r"(?i)transfer|neft|rtgs|imps|upi/": TransactionCategory.TRANSFER,
        }

    def classify_transaction(self, transaction: Transaction) -> Transaction:
        """
        Classify a single transaction using a multi-step approach.

        Args:
            transaction: The transaction to classify.

        Returns:
            The classified transaction with updated category and confidence.
        """
        # Step 1: Try rule-based classification
        rule_match = self.apply_rules(transaction)
        if rule_match and rule_match[1] > self.confidence_threshold:
            transaction.category = rule_match[0]
            transaction.confidence = rule_match[1]
            return transaction

        # Step 2: Use LLM for complex classification
        llm_classification = self.classify_with_llm(transaction)
        transaction.category = llm_classification[0]
        transaction.confidence = llm_classification[1]

        return transaction

    def apply_rules(self, transaction: Transaction) -> Optional[Tuple[TransactionCategory, float]]:
        """
        Apply rule-based classification to a transaction.

        Args:
            transaction: The transaction to classify.

        Returns:
            A tuple of (category, confidence) if a rule matches, None otherwise.
        """
        import re

        description = transaction.description.lower()

        for pattern, category in self.rules.items():
            if re.search(pattern, description):
                # Simple confidence score based on the match length relative to description
                match = re.search(pattern, description)
                if match:
                    match_length = match.end() - match.start()
                    confidence = min(0.9, 0.5 + (match_length / len(description)))
                    return (category, confidence)

        return None

    def classify_with_llm(self, transaction: Transaction) -> Tuple[TransactionCategory, float]:
        """
        Use LLM to classify a transaction.

        Args:
            transaction: The transaction to classify.

        Returns:
            A tuple of (category, confidence).
        """
        # If LLM is not available, return uncategorized with low confidence
        if not hasattr(self, 'has_llm') or not self.has_llm:
            logger.info("LLM not available. Skipping LLM classification.")
            return (TransactionCategory.UNCATEGORIZED, 0.5)

        try:
            # Prepare the prompt for the LLM
            prompt = f"""
            You are a financial transaction classifier for Indian businesses. Classify the following transaction into the most appropriate category.

            Transaction Details:
            - Date: {transaction.date.strftime('%Y-%m-%d')}
            - Description: {transaction.description}
            - Amount: ₹{transaction.amount}
            - Type: {transaction.type.value}

            Industry Context: {self.industry}

            Available Categories:
            {json.dumps({c.name: c.value for c in TransactionCategory}, indent=2)}

            Respond with a JSON object containing:
            1. "category": The category value (not name) that best matches this transaction
            2. "confidence": A number between 0 and 1 indicating your confidence in this classification
            3. "explanation": A brief explanation of why you chose this category

            JSON Response:
            """

            # Call the GROQ API with Llama model
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",  # Using Llama 3 8B model
                messages=[
                    {"role": "system", "content": "You are a financial transaction classifier for Indian businesses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )

            # Parse the response
            result = json.loads(response.choices[0].message.content)

            # Validate and return the classification
            category_value = result.get("category")
            confidence = float(result.get("confidence", 0.7))

            # Ensure the category is valid
            try:
                # Try exact match first
                category = TransactionCategory(category_value)
            except ValueError:
                # If not an exact match, try to find the closest match
                logger.warning(f"Invalid category returned by LLM: {category_value}")

                # Map common category names to our enum values
                category_map = {
                    "EXPENSE_OFFICE_SUPPLIES": TransactionCategory.EXPENSE_OFFICE_SUPPLIES,
                    "EXPENSE_UTILITIES": TransactionCategory.EXPENSE_UTILITIES,
                    "EXPENSE_SALARY": TransactionCategory.EXPENSE_SALARY,
                    "EXPENSE_RENT": TransactionCategory.EXPENSE_RENT,
                    "EXPENSE_TRAVEL": TransactionCategory.EXPENSE_TRAVEL,
                    "EXPENSE_MEALS": TransactionCategory.EXPENSE_MEALS,
                    "EXPENSE_ADVERTISING": TransactionCategory.EXPENSE_ADVERTISING,
                    "EXPENSE_PROFESSIONAL_SERVICES": TransactionCategory.EXPENSE_PROFESSIONAL_SERVICES,
                    "EXPENSE_INSURANCE": TransactionCategory.EXPENSE_INSURANCE,
                    "EXPENSE_MAINTENANCE": TransactionCategory.EXPENSE_MAINTENANCE,
                    "EXPENSE_OTHER": TransactionCategory.EXPENSE_OTHER,
                    "INCOME_BUSINESS": TransactionCategory.INCOME_BUSINESS,
                    "INCOME_INTEREST": TransactionCategory.INCOME_INTEREST,
                    "INCOME_RENT": TransactionCategory.INCOME_RENT,
                    "INCOME_OTHER": TransactionCategory.INCOME_OTHER,
                    "TAX_GST": TransactionCategory.TAX_GST,
                    "TAX_INCOME": TransactionCategory.TAX_INCOME,
                    "TAX_TDS": TransactionCategory.TAX_TDS,
                    "TAX_OTHER": TransactionCategory.TAX_OTHER,
                    "TRANSFER": TransactionCategory.TRANSFER,
                    "PERSONAL": TransactionCategory.PERSONAL,
                }

                # Check if the category is in our map
                if category_value in category_map:
                    category = category_map[category_value]
                    logger.info(f"Mapped '{category_value}' to {category.value}")
                else:
                    # If still not found, use UNCATEGORIZED
                    category = TransactionCategory.UNCATEGORIZED
                    confidence = 0.5

            return (category, confidence)

        except Exception as e:
            logger.error(f"Error classifying transaction with LLM: {e}")
            return (TransactionCategory.UNCATEGORIZED, 0.5)

    def classify_batch(self, transactions: List[Transaction]) -> List[Transaction]:
        """
        Classify a batch of transactions.

        Args:
            transactions: List of transactions to classify.

        Returns:
            List of classified transactions.
        """
        classified_transactions = []

        # Add rate limiting to avoid hitting API limits
        delay_seconds = 0.5  # Start with a small delay
        max_delay = 5.0      # Maximum delay between requests
        consecutive_errors = 0

        for i, transaction in enumerate(transactions):
            try:
                # Apply rate limiting
                if i > 0 and hasattr(self, 'has_llm') and self.has_llm:
                    time.sleep(delay_seconds)

                # Classify the transaction
                classified_transaction = self.classify_transaction(transaction)
                classified_transactions.append(classified_transaction)

                # If successful, gradually reduce delay if we had increased it
                if consecutive_errors > 0:
                    consecutive_errors = 0
                    delay_seconds = max(0.5, delay_seconds * 0.8)  # Gradually reduce delay

                # Log progress for large batches
                if (i+1) % 10 == 0 or i+1 == len(transactions):
                    logger.info(f"Classified {i+1}/{len(transactions)} transactions")

            except Exception as e:
                logger.error(f"Error classifying transaction {i+1}/{len(transactions)}: {e}")

                # Handle rate limiting errors by increasing delay
                if "429" in str(e) or "rate limit" in str(e).lower():
                    consecutive_errors += 1
                    delay_seconds = min(max_delay, delay_seconds * 2)  # Exponential backoff
                    logger.warning(f"Rate limit hit. Increasing delay to {delay_seconds} seconds")
                    time.sleep(delay_seconds * 2)  # Wait longer after a rate limit error

                # Still add the transaction, but mark it as uncategorized
                transaction.category = TransactionCategory.UNCATEGORIZED
                transaction.confidence = 0.0
                classified_transactions.append(transaction)

        return classified_transactions
