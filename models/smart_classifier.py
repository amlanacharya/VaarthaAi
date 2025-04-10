import os
import json
import re
import time
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Optional imports for BERT (will be used if available)
try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.nn.functional import softmax
    from sklearn.preprocessing import LabelEncoder
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

# Optional import for GROQ (will be used if available)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from models.transaction import Transaction, TransactionType, TransactionCategory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Coworking venue specific categories
COWORKING_INCOME_CATEGORIES = [
    TransactionCategory.INCOME_BUSINESS,    # Memberships, hot desks, private offices
    TransactionCategory.INCOME_RENT,        # Meeting room rentals, event space
    TransactionCategory.INCOME_OTHER        # Services, deposits, other income
]

COWORKING_EXPENSE_CATEGORIES = [
    TransactionCategory.EXPENSE_RENT,       # Rent for the space
    TransactionCategory.EXPENSE_UTILITIES,  # Electricity, water, internet
    TransactionCategory.EXPENSE_MAINTENANCE,# Cleaning, repairs, equipment maintenance
    TransactionCategory.EXPENSE_SALARY,     # Staff salaries
    TransactionCategory.EXPENSE_OFFICE_SUPPLIES, # Coffee, tea, office supplies
    TransactionCategory.EXPENSE_ADVERTISING,# Marketing and advertising
    TransactionCategory.EXPENSE_PROFESSIONAL_SERVICES, # Accounting, legal
    TransactionCategory.EXPENSE_INSURANCE,  # Property insurance
    TransactionCategory.EXPENSE_OTHER       # Miscellaneous expenses
]

# Define keywords for each category
CATEGORY_KEYWORDS = {
    # Income Keywords
    TransactionCategory.INCOME_BUSINESS.value: [
        'membership', 'subscription', 'hot desk', 'dedicated desk', 
        'private cabin', 'office', 'coworking', 'workspace'
    ],
    TransactionCategory.INCOME_RENT.value: [
        'conference room', 'meeting room', 'board room',
        'event space', 'training room', 'rental'
    ],
    TransactionCategory.INCOME_OTHER.value: [
        'printing', 'scanning', 'coffee', 'locker',
        'event ticket', 'catering', 'business address',
        'partnership', 'sponsor', 'commission'
    ],
    
    # Expense Keywords
    TransactionCategory.EXPENSE_RENT.value: [
        'rent', 'lease', 'property', 'landlord'
    ],
    TransactionCategory.EXPENSE_UTILITIES.value: [
        'electricity', 'water', 'internet', 'wifi', 
        'broadband', 'utility', 'phone', 'telecom'
    ],
    TransactionCategory.EXPENSE_MAINTENANCE.value: [
        'repair', 'cleaning', 'plumbing', 'electrical',
        'air conditioning', 'ac', 'maintenance', 'pest control'
    ],
    TransactionCategory.EXPENSE_SALARY.value: [
        'salary', 'payroll', 'wages', 'staff', 'employee',
        'compensation', 'bonus', 'commission'
    ],
    TransactionCategory.EXPENSE_OFFICE_SUPPLIES.value: [
        'coffee', 'tea', 'pantry', 'stationery', 'office supplies',
        'paper', 'printer', 'toner', 'furniture', 'equipment'
    ],
    TransactionCategory.EXPENSE_ADVERTISING.value: [
        'advertising', 'promotion', 'social media', 'marketing',
        'branding', 'campaign', 'publicity', 'sponsorship'
    ],
    TransactionCategory.EXPENSE_PROFESSIONAL_SERVICES.value: [
        'accounting', 'legal', 'consultant', 'advisor',
        'professional', 'service', 'audit', 'tax'
    ],
    TransactionCategory.EXPENSE_INSURANCE.value: [
        'insurance', 'policy', 'premium', 'coverage',
        'liability', 'property insurance'
    ],
    TransactionCategory.EXPENSE_OTHER.value: [
        'bank charges', 'fee', 'tax', 'travel', 'training',
        'subscription', 'software', 'miscellaneous'
    ]
}

# Regex patterns for common transaction types
REGEX_PATTERNS = {
    TransactionCategory.INCOME_BUSINESS.value: [
        r'(?i)membership|subscription|coworking|workspace',
        r'(?i)payment\s+received|invoice\s+payment'
    ],
    TransactionCategory.EXPENSE_RENT.value: [
        r'(?i)rent\s+payment|lease\s+payment',
        r'(?i)to\s+landlord|property\s+owner'
    ],
    TransactionCategory.EXPENSE_UTILITIES.value: [
        r'(?i)electricity\s+bill|water\s+bill|internet\s+bill',
        r'(?i)utility\s+payment|broadband'
    ],
    TransactionCategory.EXPENSE_SALARY.value: [
        r'(?i)salary\s+payment|payroll|wages',
        r'(?i)staff\s+payment|employee\s+compensation'
    ]
}

class SmartTransactionClassifier:
    """
    Smart transaction classifier that combines multiple classification methods
    to reduce the need for API calls.
    """
    
    def __init__(self, industry="coworking"):
        """
        Initialize the smart transaction classifier.
        
        Args:
            industry: The industry context for classification.
        """
        self.industry = industry
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.transaction_cache = {}  # Cache for previously classified transactions
        self.similarity_threshold = 85  # Threshold for fuzzy matching
        
        # Initialize GROQ client if API key is available
        if self.groq_api_key and GROQ_AVAILABLE:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                self.has_groq = True
                logger.info("GROQ client initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize GROQ client: {e}")
                self.has_groq = False
        else:
            logger.warning("GROQ API not available. Using local classification only.")
            self.has_groq = False
        
        # Initialize ML components
        self.ml_model = None
        self.vectorizer = None
        self.load_ml_model()
        
        # Initialize BERT if available
        self.bert_labeler = None
        if BERT_AVAILABLE:
            try:
                self.bert_labeler = BERTLabeler()
                self.use_bert = True
                logger.info("BERT labeler initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize BERT labeler: {e}")
                self.use_bert = False
        else:
            logger.warning("BERT not available. Skipping BERT initialization.")
            self.use_bert = False
    
    def load_ml_model(self):
        """Load the ML model and vectorizer from disk if available."""
        try:
            if os.path.exists('transaction_model.pkl'):
                with open('transaction_model.pkl', 'rb') as f:
                    self.ml_model = pickle.load(f)
                with open('vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("Loaded ML model and vectorizer from disk")
            else:
                self.ml_model = MultinomialNB()
                self.vectorizer = TfidfVectorizer()
                logger.info("Initialized new ML model and vectorizer")
        except Exception as e:
            logger.warning(f"Error loading ML model: {e}")
            self.ml_model = MultinomialNB()
            self.vectorizer = TfidfVectorizer()
    
    def save_ml_model(self):
        """Save the ML model and vectorizer to disk."""
        try:
            with open('transaction_model.pkl', 'wb') as f:
                pickle.dump(self.ml_model, f)
            with open('vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            logger.info("Saved ML model and vectorizer to disk")
        except Exception as e:
            logger.warning(f"Error saving ML model: {e}")
    
    def clean_description(self, description):
        """Clean and normalize transaction description."""
        if not description:
            return ""
        # Convert to lowercase and remove special characters
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', description.lower())
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def check_cache(self, description):
        """Check if a similar transaction exists in the cache."""
        if not self.transaction_cache:
            return None, 0
        
        # Try exact match first
        if description in self.transaction_cache:
            return self.transaction_cache[description], 100
        
        # Try fuzzy matching
        closest_match, score = process.extractOne(
            description,
            self.transaction_cache.keys(),
            scorer=fuzz.token_sort_ratio
        )
        
        if score >= self.similarity_threshold:
            return self.transaction_cache[closest_match], score
        
        return None, 0
    
    def update_cache(self, description, category, confidence):
        """Add a transaction to the cache."""
        if description and confidence >= 70:  # Only cache transactions with decent confidence
            self.transaction_cache[description] = category
    
    def keyword_based_classification(self, description, transaction_type):
        """Classify transaction based on keywords."""
        description = description.lower()
        
        # Determine which categories to check based on transaction type
        categories = COWORKING_INCOME_CATEGORIES if transaction_type == TransactionType.CREDIT else COWORKING_EXPENSE_CATEGORIES
        
        best_match = None
        best_score = 0
        
        for category in categories:
            keywords = CATEGORY_KEYWORDS.get(category.value, [])
            for keyword in keywords:
                if keyword.lower() in description:
                    # Calculate a score based on keyword length relative to description
                    score = min(85, 60 + (len(keyword) / len(description)) * 40)
                    if score > best_score:
                        best_match = category
                        best_score = score
        
        return best_match, best_score
    
    def regex_based_classification(self, description, transaction_type):
        """Classify transaction based on regex patterns."""
        description = description.upper()
        
        # Determine which patterns to check based on transaction type
        categories = COWORKING_INCOME_CATEGORIES if transaction_type == TransactionType.CREDIT else COWORKING_EXPENSE_CATEGORIES
        
        for category in categories:
            patterns = REGEX_PATTERNS.get(category.value, [])
            for pattern in patterns:
                if re.search(pattern, description, re.IGNORECASE):
                    return category, 90  # High confidence for regex matches
        
        return None, 0
    
    def ml_based_classification(self, description):
        """Classify transaction using the ML model."""
        if self.ml_model is None or self.vectorizer is None:
            return None, 0
        
        try:
            # Transform the description
            X = self.vectorizer.transform([description])
            
            # Get prediction and probability
            prediction = self.ml_model.predict(X)[0]
            proba = self.ml_model.predict_proba(X)[0]
            confidence = proba.max() * 100
            
            # Convert prediction to category
            for category in list(TransactionCategory):
                if category.value == prediction:
                    return category, confidence
            
            return None, 0
        except Exception as e:
            logger.warning(f"Error in ML classification: {e}")
            return None, 0
    
    def bert_based_classification(self, description):
        """Classify transaction using BERT if available."""
        if not self.use_bert or self.bert_labeler is None:
            return None, 0
        
        try:
            category_value, confidence = self.bert_labeler.predict(description)
            
            # Convert category value to enum
            for category in list(TransactionCategory):
                if category.value == category_value:
                    return category, confidence
            
            return None, 0
        except Exception as e:
            logger.warning(f"Error in BERT classification: {e}")
            return None, 0
    
    def groq_based_classification(self, transaction):
        """Classify transaction using GROQ API."""
        if not self.has_groq:
            return None, 0
        
        try:
            # Prepare the prompt for the LLM
            prompt = f"""
            You are a financial transaction classifier for coworking space businesses. Classify the following transaction into the most appropriate category.
            
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
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",  # Using Llama 3 8B model
                messages=[
                    {"role": "system", "content": "You are a financial transaction classifier for coworking space businesses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Validate and return the classification
            category_value = result.get("category")
            confidence = float(result.get("confidence", 0.7)) * 100
            
            # Convert category value to enum
            for category in list(TransactionCategory):
                if category.value == category_value:
                    return category, confidence
            
            # If category not found, map common category names
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
            
            if category_value in category_map:
                return category_map[category_value], confidence
            
            return TransactionCategory.UNCATEGORIZED, 50
            
        except Exception as e:
            logger.warning(f"Error in GROQ classification: {e}")
            return None, 0
    
    def train_ml_model(self, descriptions, categories):
        """Train the ML model with new data."""
        if not descriptions or not categories:
            return False
        
        try:
            # Convert categories to values if they are enums
            category_values = []
            for category in categories:
                if isinstance(category, TransactionCategory):
                    category_values.append(category.value)
                else:
                    category_values.append(category)
            
            # Fit the vectorizer and transform descriptions
            X = self.vectorizer.fit_transform(descriptions)
            
            # Train the model
            self.ml_model.fit(X, category_values)
            
            # Save the model
            self.save_ml_model()
            
            logger.info(f"Trained ML model with {len(descriptions)} examples")
            return True
        except Exception as e:
            logger.warning(f"Error training ML model: {e}")
            return False
    
    def classify_transaction(self, transaction):
        """
        Classify a transaction using multiple methods in order of efficiency.
        
        Args:
            transaction: The transaction to classify.
            
        Returns:
            The classified transaction with updated category and confidence.
        """
        # Clean the description
        cleaned_description = self.clean_description(transaction.description)
        
        # Step 1: Check cache for similar transactions
        cached_category, cache_confidence = self.check_cache(cleaned_description)
        if cached_category:
            transaction.category = cached_category
            transaction.confidence = cache_confidence
            return transaction
        
        # Step 2: Try regex-based classification
        regex_category, regex_confidence = self.regex_based_classification(
            transaction.description, transaction.type
        )
        if regex_category and regex_confidence >= 85:
            transaction.category = regex_category
            transaction.confidence = regex_confidence
            self.update_cache(cleaned_description, regex_category, regex_confidence)
            return transaction
        
        # Step 3: Try keyword-based classification
        keyword_category, keyword_confidence = self.keyword_based_classification(
            transaction.description, transaction.type
        )
        if keyword_category and keyword_confidence >= 80:
            transaction.category = keyword_category
            transaction.confidence = keyword_confidence
            self.update_cache(cleaned_description, keyword_category, keyword_confidence)
            return transaction
        
        # Step 4: Try ML-based classification if available
        ml_category, ml_confidence = self.ml_based_classification(cleaned_description)
        if ml_category and ml_confidence >= 75:
            transaction.category = ml_category
            transaction.confidence = ml_confidence
            self.update_cache(cleaned_description, ml_category, ml_confidence)
            return transaction
        
        # Step 5: Try BERT-based classification if available
        if self.use_bert:
            bert_category, bert_confidence = self.bert_based_classification(cleaned_description)
            if bert_category and bert_confidence >= 80:
                transaction.category = bert_category
                transaction.confidence = bert_confidence
                self.update_cache(cleaned_description, bert_category, bert_confidence)
                return transaction
        
        # Step 6: Use GROQ API as a last resort
        if self.has_groq:
            groq_category, groq_confidence = self.groq_based_classification(transaction)
            if groq_category:
                transaction.category = groq_category
                transaction.confidence = groq_confidence
                self.update_cache(cleaned_description, groq_category, groq_confidence)
                return transaction
        
        # If all methods fail, use the best available result
        best_category = None
        best_confidence = 0
        
        for category, confidence in [
            (regex_category, regex_confidence),
            (keyword_category, keyword_confidence),
            (ml_category, ml_confidence),
            (bert_category if self.use_bert else None, bert_confidence if self.use_bert else 0)
        ]:
            if category and confidence > best_confidence:
                best_category = category
                best_confidence = confidence
        
        if best_category:
            transaction.category = best_category
            transaction.confidence = best_confidence
        else:
            # Default to uncategorized with low confidence
            transaction.category = TransactionCategory.UNCATEGORIZED
            transaction.confidence = 30
        
        return transaction
    
    def classify_batch(self, transactions):
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
        api_calls = 0
        
        for i, transaction in enumerate(transactions):
            try:
                # Apply rate limiting only if we're using GROQ
                if i > 0 and self.has_groq and api_calls > 0:
                    time.sleep(delay_seconds)
                
                # Classify the transaction
                start_time = time.time()
                classified_transaction = self.classify_transaction(transaction)
                end_time = time.time()
                
                # Check if GROQ API was used (based on timing)
                if end_time - start_time > 0.5 and self.has_groq:
                    api_calls += 1
                
                classified_transactions.append(classified_transaction)
                
                # If successful, gradually reduce delay if we had increased it
                if consecutive_errors > 0:
                    consecutive_errors = 0
                    delay_seconds = max(0.5, delay_seconds * 0.8)  # Gradually reduce delay
                
                # Log progress for large batches
                if (i+1) % 10 == 0 or i+1 == len(transactions):
                    logger.info(f"Classified {i+1}/{len(transactions)} transactions (API calls: {api_calls})")
                    
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
        
        # After classifying all transactions, train the ML model with the results
        self._update_ml_model(classified_transactions)
        
        logger.info(f"Classification complete. Made {api_calls} API calls for {len(transactions)} transactions.")
        return classified_transactions
    
    def _update_ml_model(self, transactions):
        """Update the ML model with newly classified transactions."""
        # Only use transactions with high confidence
        high_confidence_transactions = [
            t for t in transactions if t.confidence >= 80
        ]
        
        if len(high_confidence_transactions) >= 10:
            descriptions = [self.clean_description(t.description) for t in high_confidence_transactions]
            categories = [t.category for t in high_confidence_transactions]
            
            self.train_ml_model(descriptions, categories)


class BERTLabeler:
    """BERT-based transaction classifier."""
    
    def __init__(self, model_path='transaction_bert'):
        """
        Initialize the BERT labeler.
        
        Args:
            model_path: Path to save/load the BERT model.
        """
        if not BERT_AVAILABLE:
            raise ImportError("BERT dependencies not available")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_encoder = LabelEncoder()
        self.model = None
        self.model_path = model_path
        
        # Fit label encoder with all possible categories
        all_categories = [c.value for c in TransactionCategory]
        self.label_encoder.fit(all_categories)
        
        self.load_model()
    
    def load_model(self):
        """Load the BERT model from disk if available."""
        try:
            if os.path.exists(self.model_path):
                self.model = BertForSequenceClassification.from_pretrained(self.model_path)
                self.model.to(self.device)
                logger.info("Loaded existing BERT model")
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=len(self.label_encoder.classes_)
                )
                self.model.to(self.device)
                logger.info("Initialized new BERT model")
        except Exception as e:
            logger.warning(f"Error loading BERT model: {e}")
            self.model = None
    
    def save_model(self):
        """Save the BERT model to disk."""
        if self.model:
            self.model.save_pretrained(self.model_path)
            logger.info("Saved BERT model")
    
    def predict(self, text):
        """
        Predict the category of a transaction description.
        
        Args:
            text: The transaction description.
            
        Returns:
            A tuple of (category, confidence).
        """
        if not self.model:
            return None, 0
        
        try:
            # Prepare input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1)
                confidence = float(torch.max(probs))
            
            # Convert prediction to label
            predicted_label = self.label_encoder.inverse_transform([prediction.item()])[0]
            return predicted_label, confidence * 100
        except Exception as e:
            logger.warning(f"Error in BERT prediction: {e}")
            return None, 0
    
    def train(self, texts, labels, epochs=3):
        """
        Train the BERT model with new data.
        
        Args:
            texts: List of transaction descriptions.
            labels: List of category values.
            epochs: Number of training epochs.
            
        Returns:
            True if training was successful, False otherwise.
        """
        if not texts or not labels:
            return False
        
        try:
            # Convert labels to values if they are enums
            label_values = []
            for label in labels:
                if isinstance(label, TransactionCategory):
                    label_values.append(label.value)
                else:
                    label_values.append(label)
            
            # Encode labels
            encoded_labels = self.label_encoder.transform(label_values)
            
            # Prepare dataset
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            labels_tensor = torch.tensor(encoded_labels).to(self.device)
            
            # Training settings
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.model(**inputs, labels=labels_tensor)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
            self.save_model()
            return True
        except Exception as e:
            logger.warning(f"Error training BERT model: {e}")
            return False
