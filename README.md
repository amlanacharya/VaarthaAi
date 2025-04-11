# VaarthaAI - Financial Assistant for Indian CAs and Businesses

VaarthaAI is an AI-powered financial assistant that helps Indian Chartered Accountants (CAs) and businesses streamline financial workflows, improve compliance, and boost productivity. This application uses modern open-source technologies including GROQ API with Llama models and local embeddings for efficient transaction classification and financial insights.

## Features

- **Smart Transaction Classification**: Multi-layered classification system that minimizes API calls
- **Financial Insights**: AI-driven insights based on transaction patterns
- **Compliance Assistant**: GST reconciliation and tax deduction identification
- **Natural Language Interface**: Ask questions about financial data and Indian regulations

## Project Structure

```
VaarthaAI/
├── .gitignore               # Git ignore file, used to exclude files/folders from version control
├── README.md                # Project documentation file (overview and setup instructions)
├── RUNNING.md               # Instructions on how to run the project
├── Vaartha.docx             # doc version of plan
├── app                      # Main application folder
│   ├── __init__.py          # Marks the directory as a Python package
│   ├── app.py               # Main application logic file
│   └── run.py               # Entry point to run the app
├── batch                    # Folder containing batch script files for running or testing
│   ├── patch_streamlit.bat  # Batch script for patching or updating Streamlit
│   ├── reset_app.bat        # Batch script for resetting the app
│   ├── run_fixed_app.bat    # Batch script for running the fixed version of the app
│   ├── run_no_watcher.bat   # Batch script to run the app without a watcher
│   ├── run_smart_app.bat    # Batch script for running the smart app
│   ├── run_smart_classifier.bat # Batch script for running the smart classifier
│   ├── run_smart_classifier_no_groq.bat # Batch script for running classifier without Groq
│   └── test_simple.bat      # Batch script for running simple tests
├── check.py                 # Python script for performing checks or validations
├── cli.py                   # Command-line interface script for interacting with the app
├── config.py                # Configuration file (settings, constants, etc.)
├── controllers.py           # Python file for handling controller logic
├── data                     # Folder for data files
│   ├── chroma_db            # Folder containing Chroma DB-related files
│   │ 
│   ├── regulations           # Folder containing regulation-related JSON files
│   ├── sample_data           # Folder containing sample data files (CSV)
│   |
│   ├── transaction_query_vectorstore # Folder containing transaction query data
│   │  
│   └── vaartha.db            # Database file 
├── exceptions.py             # Custom exception classes for the project
├── main.py                   # Main script to run the project
├── models                    # Folder business logic models
│   ├── __init__.py            # Marks the directory as a Python package
│   ├── classifier.py          # Machine learning classifier model
│   ├── rag.py                 # RAG (Retrieval-Augmented Generation)
│   ├── smart_classifier.py    # Smart classifier model
│   ├── transaction.py         # Model for handling transaction-related logic
│   └── unified_query.py       # Model for managing unified queries
├── patch_streamlit.py        # Python script for patching Streamlit
├── plan.md                   # Project planning document (roadmap, goals, etc.)
├── requirements.txt          # List of dependencies (used by pip for installations)
├── run_streamlit.py          # Script to run the Streamlit app
├── sql_loader.py             # Python script for loading SQL data
├── test                      # Folder for test scripts
├── transaction_model.pkl     # Pickled machine learning model for transactions
├── transaction_nl_query.py   # Script for handling transaction-related natural language queries
├── utils                     # Folder for utility scripts
│   ├── __init__.py            # Marks the directory as a Python package
│   ├── data_query.py          # Utility script for querying data
│   ├── database.py            # Utility script for database-related logic
│   ├── parser.py              # Parser utility for processing data
│   ├── patch.py               # Utility script for applying patches
│   ├── reset.py               # Utility script for resetting configurations or data
│   └── sqlite_fix.py          # Utility script for fixing SQLite-related issues
├── vaartha.py                # Main script for the Vaartha project 
└── vectorizer.pkl            # Pickled vectorizer model (likely used for text processing)


```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/VaarthaAI.git
   cd VaarthaAI
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root with the following:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   VAARTHA_ENV=development
   ```

5. Run the application:
   ```bash
   python vaartha.py run
   ```

## Usage Examples

### Run the Web Application

```bash
# Standard run
python vaartha.py run

# Run with disabled file watcher (fixes PyTorch conflict)
python vaartha.py run --no-watcher

# Run in headless mode
python vaartha.py run --headless
```

### Test the Transaction Classifier

```bash
# Test with GROQ API if available
python vaartha.py test

# Test without using GROQ API
python vaartha.py test --no-groq

# Test with a specific file
python vaartha.py test --file data/sample_data/custom_statement.csv --bank hdfc
```

### Reset Application Data

```bash
# Reset all data (with confirmation prompt)
python vaartha.py reset

# Force reset without confirmation
python vaartha.py reset --force

# Reset only the database
python vaartha.py reset --database

# Reset only the vector database
python vaartha.py reset --vector-db
```

### Generate Sample Data

```bash
# Generate default sample data
python vaartha.py sample

# Generate custom number of transactions
python vaartha.py sample --count 50

# Generate for a specific industry
python vaartha.py sample --industry retail
```

## Smart Classification System

The application uses a multi-layered approach to transaction classification:

1. **Cache-Based Classification**: Remembers previously classified transactions
2. **Rule-Based Classification**: Uses regex patterns and keywords for common transactions
3. **Machine Learning Classification**: Uses a local ML model that improves over time
4. **BERT-Based Classification**: Optional deep learning for complex cases
5. **GROQ API Classification**: Only used as a last resort for difficult transactions

This approach minimizes API calls while maintaining high accuracy.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_classifier.py

# Run tests with verbose output
pytest -v

# Run tests that don't require API access
pytest -k "not api"
```

### Project Organization

The project follows these design principles:

- **Dependency Injection**: Controllers accept dependencies for easier testing
- **Separation of Concerns**: Business logic in controllers, UI in app module
- **MVC Pattern**: Model-View-Controller architecture
- **Error Handling**: Centralized exceptions and standardized error responses
- **Configuration Management**: Centralized configuration with environment support

## Troubleshooting

### Common Issues

1. **SQLite Thread Error**: If you encounter thread-related SQLite errors, the improved database module with context managers should resolve this.

2. **PyTorch/Streamlit Conflict**: If you encounter PyTorch-related errors with Streamlit:
   ```bash
   # Apply the Streamlit patch
   python vaartha.py patch --streamlit
   
   # Or run with file watcher disabled
   python vaartha.py run --no-watcher
   ```

3. **ChromaDB Errors**: If the vector database is corrupted:
   ```bash
   python vaartha.py reset --vector-db
   ```

## License

[MIT License](LICENSE)

## Acknowledgements

- GROQ for providing Llama model API access
- Hugging Face for sentence-transformers and BERT models
- LangChain for the RAG implementation
- Streamlit for the web interface
- Other open source contributors
