# VaarthaAI - Financial Assistant for Indian CAs and Businesses

VaarthaAI is an AI-powered financial assistant that helps Indian Chartered Accountants (CAs) and businesses streamline financial workflows, improve compliance, and boost productivity. This prototype uses open-source technologies including GROQ API with Llama models and local embeddings.

## Features

- **Smart Transaction Classification**: Efficiently categorizes bank transactions using a multi-layered approach with minimal API calls
- **Financial Insights**: Provides actionable insights based on transaction patterns
- **Compliance Assistant**: Helps with GST reconciliation and finding tax deductions
- **Natural Language Interface**: Ask questions about financial data and regulations

## Project Structure

```
VaarthaAI/
├── app/
│   └── app.py                    # Streamlit application
├── data/
│   ├── chroma_db/                # Vector database for RAG
│   ├── regulations/              # Financial regulations data
│   └── sample_data/              # Sample transaction data
├── models/
│   ├── classifier.py             # Basic transaction classifier
│   ├── smart_classifier.py       # Advanced multi-layered classifier
│   ├── rag.py                    # Retrieval Augmented Generation system
│   └── transaction.py            # Transaction data models
├── utils/
│   ├── database.py               # Thread-safe database utilities
│   └── parser.py                 # Bank statement parser
├── scripts/
│   ├── reset_app.py              # Reset application data
│   └── patch_streamlit.py        # Fix PyTorch/Streamlit conflicts
├── .env                          # Environment variables
├── run_smart_app.py              # Run app with smart classifier
├── run_smart_classifier.py       # Test the smart classifier
└── README.md                     # Project documentation
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/VaarthaAI.git
   cd VaarthaAI
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your GROQ API key: `GROQ_API_KEY=your_groq_api_key_here`

5. Run the application:
   ```
   python run_smart_app.py
   ```

   Or use one of the provided batch files:
   - `run_fixed_app.bat` - Run with thread-safe database
   - `run_smart_classifier.bat` - Test the smart classifier
   - `run_smart_classifier_no_groq.bat` - Test without GROQ API

## Usage

1. **Upload Bank Statements**: Upload bank statements in CSV, Excel, or PDF format
2. **View and Edit Transactions**: Review and modify transaction categories
3. **Get Financial Insights**: Analyze expenses by category and ask questions
4. **Compliance Assistance**: Find potential tax deductions and reconcile GST

## Development

### Smart Classification System

The application uses a multi-layered classification approach to minimize API calls:

1. **Cache-based Classification**: Remembers previously classified transactions
2. **Rule-based Classification**: Uses regex patterns and keywords for common transactions
3. **Machine Learning Classification**: Uses a local ML model that improves over time
4. **BERT-based Classification**: Optional deep learning for complex cases
5. **GROQ API Classification**: Only used as a last resort for difficult transactions

### Future Enhancements

Planned improvements include:

- Integration with accounting software
- Advanced GST reconciliation
- Multi-user support for CA firms
- Mobile application

## License

[MIT License](LICENSE)

## Troubleshooting

### Common Issues

1. **SQLite Thread Error**: If you encounter `SQLite objects created in a thread can only be used in that same thread`, use the thread-safe version by running `run_fixed_app.bat`.

2. **PyTorch/Streamlit Conflict**: If you see `RuntimeError: Tried to instantiate class '__path__._path'`, run the patch script with `patch_streamlit.bat` or use `run_no_watcher.py`.

3. **ChromaDB Errors**: If the vector database is corrupted, reset it with `reset_app.bat`.

## Acknowledgements

- GROQ for Llama model API access
- Hugging Face for sentence-transformers and BERT models
- LangChain for RAG implementation
- Streamlit for the web interface
- FuzzyWuzzy for string matching
- scikit-learn for machine learning components
