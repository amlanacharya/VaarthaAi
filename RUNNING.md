# Running VaarthaAI

This document provides instructions for running the VaarthaAI prototype with different configurations.

## Prerequisites

1. Python 3.9+ installed
2. Virtual environment created and activated
3. Dependencies installed: `pip install -r requirements.txt`

## Configuration Options

### Option 1: With GROQ API (Recommended)

For the full experience with LLM-powered transaction classification:

1. Get a GROQ API key from [groq.com](https://console.groq.com/)
2. Add your API key to the `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
3. Run the application:
   ```
   python run_streamlit.py
   ```

### Option 2: Rule-Based Only (No API Key)

If you don't have a GROQ API key, the application will fall back to rule-based classification only:

1. Run the application without setting the GROQ_API_KEY:
   ```
   python run_streamlit.py
   ```
2. The system will log a warning and use only rule-based classification for transactions.

## Testing with Sample Data

To test the transaction classifier with sample coworking venue data:

```
python test_classifier.py
```

This will:
1. Parse the sample HDFC bank statement
2. Classify transactions using rules and LLM (if available)
3. Display the results with categories and confidence scores

## Sample Data

The following sample data files are included:

- `data/sample_data/coworking_bank_statement.csv` - Basic bank statement
- `data/sample_data/coworking_detailed_statement.csv` - Detailed statement with pre-categorized transactions
- `data/sample_data/hdfc_coworking_statement.csv` - HDFC-formatted statement for testing

## Troubleshooting

If you encounter any issues:

1. **Dependency Warnings**: Update the dependencies with:
   ```
   pip install -r requirements.txt
   ```

2. **Runtime Errors**: Try using the alternative script:
   ```
   python run_streamlit.py
   ```

3. **LLM Not Working**: Check your GROQ API key and internet connection. The system will fall back to rule-based classification if LLM is unavailable.
