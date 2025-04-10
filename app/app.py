import os
import streamlit as st
import pandas as pd
from datetime import datetime
import json
import uuid
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from models.transaction import Transaction, TransactionType, TransactionCategory
from models.smart_classifier import SmartTransactionClassifier
from models.rag import FinancialRAG
from utils.parser import BankStatementParser, generate_sample_transactions
from utils.database import Database

# Configure page
st.set_page_config(
    page_title="VaarthaAI - Financial Assistant",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = None
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False

# Initialize database
@st.cache_resource
def get_database():
    return Database()

# Initialize classifier
@st.cache_resource
def get_classifier():
    return SmartTransactionClassifier(industry="coworking")

# Initialize RAG system
@st.cache_resource
def get_rag():
    rag = FinancialRAG()
    if not st.session_state.rag_initialized:
        rag.initialize_knowledge_base()
        st.session_state.rag_initialized = True
    return rag

# Header
st.title("VaarthaAI - Financial Assistant")
st.subheader("AI-powered financial assistant for Indian CAs and businesses")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Transaction Management", "Financial Insights", "Compliance Assistant"])

# Initialize components
db = get_database()
classifier = get_classifier()
rag = get_rag()

# Ensure database is initialized
if not st.session_state.db_initialized:
    # Load transactions from database
    st.session_state.transactions = db.get_transactions(limit=100)
    st.session_state.db_initialized = True

# Dashboard Page
if page == "Dashboard":
    st.header("Dashboard")

    # Key metrics
    col1, col2, col3 = st.columns(3)

    # Calculate metrics
    total_transactions = len(st.session_state.transactions)
    total_income = sum(t.amount for t in st.session_state.transactions if t.type == TransactionType.CREDIT)
    total_expenses = sum(t.amount for t in st.session_state.transactions if t.type == TransactionType.DEBIT)

    with col1:
        st.metric("Total Transactions", total_transactions)

    with col2:
        st.metric("Total Income", f"₹{total_income:,.2f}")

    with col3:
        st.metric("Total Expenses", f"₹{total_expenses:,.2f}")

    # Recent transactions
    st.subheader("Recent Transactions")
    if st.session_state.transactions:
        # Convert to DataFrame for display
        df = pd.DataFrame([t.to_dict() for t in st.session_state.transactions[:10]])

        # Format date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # Format amount with currency symbol
        if 'amount' in df.columns:
            df['amount'] = df['amount'].apply(lambda x: f"₹{x:,.2f}")

        # Select columns to display
        display_cols = ['date', 'description', 'amount', 'type', 'category']
        df = df[display_cols]

        # Rename columns for better display
        df.columns = [col.capitalize() for col in df.columns]

        st.dataframe(df, use_container_width=True)
    else:
        st.info("No transactions found. Upload a bank statement or generate sample data.")

        if st.button("Generate Sample Data"):
            # Generate sample transactions
            sample_transactions = generate_sample_transactions(20)

            # Classify the transactions
            classified_transactions = classifier.classify_batch(sample_transactions)

            # Save to database
            for transaction in classified_transactions:
                db.save_transaction(transaction)

            # Update session state
            st.session_state.transactions = classified_transactions
            st.success("Generated and classified 20 sample transactions!")
            st.rerun()

# Transaction Management Page
elif page == "Transaction Management":
    st.header("Transaction Management")

    # Tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Upload Statement", "View Transactions", "Edit Categories"])

    # Upload Statement Tab
    with tab1:
        st.subheader("Upload Bank Statement")

        uploaded_file = st.file_uploader("Choose a bank statement file", type=["csv", "xlsx", "pdf"])
        bank_name = st.selectbox("Bank Name", ["HDFC", "SBI", "ICICI", "Axis", "Other"])

        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            st.info(f"Processing {uploaded_file.name}...")

            if st.button("Process Statement"):
                try:
                    # Parse the bank statement
                    parser = BankStatementParser()
                    batch = parser.parse(tmp_path, bank_name.lower())

                    if not batch.transactions:
                        st.warning("No transactions found in the statement. Try a different format or bank.")
                    else:
                        # Classify the transactions
                        batch.transactions = classifier.classify_batch(batch.transactions)

                        # Save to database
                        batch_id = db.save_batch(batch)

                        # Update session state
                        st.session_state.transactions = db.get_transactions(limit=100)
                        st.session_state.current_batch = batch

                        st.success(f"Successfully processed {len(batch.transactions)} transactions!")

                except Exception as e:
                    st.error(f"Error processing statement: {str(e)}")
                finally:
                    # Clean up the temporary file
                    os.unlink(tmp_path)

    # View Transactions Tab
    with tab2:
        st.subheader("View Transactions")

        # Filter options
        col1, col2 = st.columns(2)

        with col1:
            filter_category = st.selectbox(
                "Filter by Category",
                ["All"] + [c.value for c in TransactionCategory]
            )

        with col2:
            filter_type = st.selectbox(
                "Filter by Type",
                ["All", TransactionType.CREDIT.value, TransactionType.DEBIT.value]
            )

        # Apply filters
        filtered_transactions = st.session_state.transactions

        if filter_category != "All":
            filtered_transactions = [t for t in filtered_transactions if t.category.value == filter_category]

        if filter_type != "All":
            filtered_transactions = [t for t in filtered_transactions if t.type.value == filter_type]

        # Display transactions
        if filtered_transactions:
            # Convert to DataFrame for display
            df = pd.DataFrame([t.to_dict() for t in filtered_transactions])

            # Format date
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # Format amount with currency symbol
            if 'amount' in df.columns:
                df['amount'] = df['amount'].apply(lambda x: f"₹{x:,.2f}")

            # Select columns to display
            display_cols = ['date', 'description', 'amount', 'type', 'category', 'confidence']
            df = df[display_cols]

            # Rename columns for better display
            df.columns = [col.capitalize() for col in df.columns]

            st.dataframe(df, use_container_width=True)

            # Export option
            if st.button("Export to CSV"):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    csv,
                    "transactions.csv",
                    "text/csv",
                    key='download-csv'
                )
        else:
            st.info("No transactions found with the selected filters.")

    # Edit Categories Tab
    with tab3:
        st.subheader("Edit Transaction Categories")

        # Select transaction to edit
        if st.session_state.transactions:
            # Create a simple display of transactions
            transaction_options = [f"{t.date.strftime('%Y-%m-%d')} | {t.description[:30]}... | ₹{t.amount:,.2f}" for t in st.session_state.transactions]
            selected_transaction_idx = st.selectbox("Select Transaction", range(len(transaction_options)), format_func=lambda x: transaction_options[x])

            if selected_transaction_idx is not None:
                transaction = st.session_state.transactions[selected_transaction_idx]

                # Display transaction details
                st.write("### Transaction Details")
                st.write(f"**Date:** {transaction.date.strftime('%Y-%m-%d')}")
                st.write(f"**Description:** {transaction.description}")
                st.write(f"**Amount:** ₹{transaction.amount:,.2f}")
                st.write(f"**Type:** {transaction.type.value}")

                # Edit category
                new_category = st.selectbox(
                    "Category",
                    [c for c in TransactionCategory],
                    index=list(TransactionCategory).index(transaction.category)
                )

                new_notes = st.text_area("Notes", transaction.notes or "")

                if st.button("Update Transaction"):
                    # Update transaction
                    transaction.category = new_category
                    transaction.notes = new_notes

                    # Save to database
                    db.save_transaction(transaction)

                    # Update session state
                    st.session_state.transactions[selected_transaction_idx] = transaction

                    st.success("Transaction updated successfully!")
        else:
            st.info("No transactions available to edit.")

# Financial Insights Page
elif page == "Financial Insights":
    st.header("Financial Insights")

    # Tabs for different insights
    tab1, tab2 = st.tabs(["Category Analysis", "Ask Financial Questions"])

    # Category Analysis Tab
    with tab1:
        st.subheader("Expense by Category")

        if st.session_state.transactions:
            # Prepare data for visualization
            expense_data = {}
            for t in st.session_state.transactions:
                if t.type == TransactionType.DEBIT:
                    category = t.category.value
                    if category not in expense_data:
                        expense_data[category] = 0
                    expense_data[category] += t.amount

            # Convert to DataFrame
            df = pd.DataFrame({
                'Category': list(expense_data.keys()),
                'Amount': list(expense_data.values())
            })

            # Sort by amount
            df = df.sort_values('Amount', ascending=False)

            # Display as bar chart
            st.bar_chart(df.set_index('Category'))

            # Display as table
            df['Amount'] = df['Amount'].apply(lambda x: f"₹{x:,.2f}")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No transaction data available for analysis.")

    # Ask Financial Questions Tab
    with tab2:
        st.subheader("Ask Financial Questions")

        question = st.text_input("Ask a question about your finances or Indian tax regulations")

        if question:
            with st.spinner("Generating answer..."):
                # Use RAG to answer the question
                response = rag.query(question)

                st.write("### Answer")
                st.write(response["response"])

                # Display sources
                if response["sources"]:
                    st.write("### Sources")
                    for i, source in enumerate(response["sources"]):
                        st.write(f"**Source {i+1}:** {source['title']} - {source['section_title']}")

# Compliance Assistant Page
elif page == "Compliance Assistant":
    st.header("Compliance Assistant")

    # Tabs for different compliance functions
    tab1, tab2 = st.tabs(["GST Reconciliation", "Tax Deduction Finder"])

    # GST Reconciliation Tab
    with tab1:
        st.subheader("GST Reconciliation")
        st.write("This feature helps reconcile your transactions with GST returns.")

        # Placeholder for GST reconciliation functionality
        st.info("GST reconciliation feature is under development.")

        # Display GST-related transactions
        gst_transactions = [t for t in st.session_state.transactions if t.category in [TransactionCategory.GST_INPUT, TransactionCategory.GST_OUTPUT, TransactionCategory.TAX_GST]]

        if gst_transactions:
            st.write("### GST-Related Transactions")

            # Convert to DataFrame for display
            df = pd.DataFrame([t.to_dict() for t in gst_transactions])

            # Format date
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # Format amount with currency symbol
            if 'amount' in df.columns:
                df['amount'] = df['amount'].apply(lambda x: f"₹{x:,.2f}")

            # Select columns to display
            display_cols = ['date', 'description', 'amount', 'category']
            df = df[display_cols]

            # Rename columns for better display
            df.columns = [col.capitalize() for col in df.columns]

            st.dataframe(df, use_container_width=True)
        else:
            st.info("No GST-related transactions found.")

    # Tax Deduction Finder Tab
    with tab2:
        st.subheader("Tax Deduction Finder")
        st.write("This feature helps identify potential tax deductions from your transactions.")

        if st.button("Find Potential Deductions"):
            with st.spinner("Analyzing transactions..."):
                # Placeholder for deduction finder functionality
                # In a real implementation, this would use more sophisticated logic
                potential_deductions = [
                    t for t in st.session_state.transactions
                    if t.type == TransactionType.DEBIT and t.category in [
                        TransactionCategory.EXPENSE_RENT,
                        TransactionCategory.EXPENSE_INSURANCE,
                        TransactionCategory.EXPENSE_PROFESSIONAL_SERVICES
                    ]
                ]

                if potential_deductions:
                    st.success(f"Found {len(potential_deductions)} potential tax deductions!")

                    # Convert to DataFrame for display
                    df = pd.DataFrame([t.to_dict() for t in potential_deductions])

                    # Format date
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

                    # Format amount with currency symbol
                    if 'amount' in df.columns:
                        df['amount'] = df['amount'].apply(lambda x: f"₹{x:,.2f}")

                    # Select columns to display
                    display_cols = ['date', 'description', 'amount', 'category']
                    df = df[display_cols]

                    # Rename columns for better display
                    df.columns = [col.capitalize() for col in df.columns]

                    st.dataframe(df, use_container_width=True)

                    # Calculate total potential deductions
                    total_deductions = sum(t.amount for t in potential_deductions)
                    st.write(f"### Total Potential Deductions: ₹{total_deductions:,.2f}")
                else:
                    st.info("No potential tax deductions found in your transactions.")

# Footer
st.markdown("---")
st.markdown("VaarthaAI - Powered by AI for Indian Financial Compliance")
