"""
Data query component for VaarthaAI.
Provides a natural language interface to transaction data with visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import json

from controllers import InsightController
from models.transaction import TransactionType, TransactionCategory
from models.unified_query import UnifiedQueryHandler

def render_data_query_interface(insight_controller: InsightController):
    """
    Render the data query interface with example questions and visualization options.
    
    Args:
        insight_controller: The insight controller instance
    """
    st.header("Ask Questions About Your Financial Data")
    
    # Display description
    st.write("""
    Ask questions about your financial data in plain language. You can ask about:
    - Transaction details and summaries
    - Income and expense analysis
    - Category breakdowns and trends
    - Specific time periods or amounts
    """)
    
    # Example questions
    with st.expander("Example Questions"):
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            st.subheader("Transaction Questions")
            examples = [
                "What was my highest expense last month?",
                "Show me all income transactions above ₹10,000",
                "How many transactions were classified as office supplies?",
                "What's my average monthly rent payment?"
            ]
            for example in examples:
                if st.button(example, key=f"example_{examples.index(example)}"):
                    st.session_state.user_query = example
        
        with example_col2:
            st.subheader("Analysis Questions")
            examples = [
                "What is my total income this year?",
                "Show me my monthly expenses by category",
                "How much did I spend on utilities compared to rent?",
                "Which month had the highest GST payments?"
            ]
            for example in examples:
                if st.button(example, key=f"example_{examples.index(example) + 10}"):
                    st.session_state.user_query = example
    
    # Initialize session state variables if they don't exist
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""
    if 'last_query_result' not in st.session_state:
        st.session_state.last_query_result = None
    
    # Query input
    user_query = st.text_input(
        "Ask a question about your financial data:",
        value=st.session_state.user_query,
        key="query_input"
    )
    
    # Process query when submitted
    if user_query:
        st.session_state.user_query = user_query  # Store in session state
        
        with st.spinner("Processing your question..."):
            # Get response from the insight controller
            response = insight_controller.query_financial_knowledge(user_query)
            st.session_state.last_query_result = response  # Store in session state
            
            # Display the natural language response
            st.success(response["response"])
            
            # Check if it's transaction data or regulation data
            if response.get("is_transaction_data", False) and "data" in response and response["data"]:
                # Create DataFrame from the result data
                result_df = pd.DataFrame(response["data"])
                
                # Check if there's data to display
                if not result_df.empty:
                    # Display the visualization options
                    display_transaction_visualization(result_df, response)
                    
                    # Show the SQL query in an expander
                    if "sql" in response:
                        with st.expander("Generated SQL Query"):
                            st.code(response["sql"], language="sql")
                    
                    # Show data table in an expander
                    with st.expander("Data Table"):
                        # Format the data before displaying
                        display_df = format_transaction_dataframe(result_df)
                        st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No data found matching your query.")
            
            # Display regulation sources if present
            elif "sources" in response and response["sources"]:
                st.subheader("Sources")
                for i, source in enumerate(response["sources"]):
                    st.write(f"**Source {i+1}:** {source['title']} - {source['section_title']}")
    
    # If there's a visualization preference in the state, apply it to the stored result
    if ('visualization_type' in st.session_state and 
        st.session_state.last_query_result is not None and 
        st.session_state.last_query_result.get("is_transaction_data", False) and
        "data" in st.session_state.last_query_result and 
        st.session_state.last_query_result["data"]):
        
        result_df = pd.DataFrame(st.session_state.last_query_result["data"])
        display_transaction_visualization(
            result_df, 
            st.session_state.last_query_result, 
            st.session_state.visualization_type
        )

def format_transaction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format a transaction dataframe for display.
    
    Args:
        df: Input DataFrame with transaction data
        
    Returns:
        Formatted DataFrame
    """
    # Make a copy to avoid modifying the original
    formatted_df = df.copy()
    
    # Format date columns
    date_columns = [col for col in formatted_df.columns if 'date' in col.lower()]
    for col in date_columns:
        if col in formatted_df.columns:
            try:
                formatted_df[col] = pd.to_datetime(formatted_df[col]).dt.strftime('%Y-%m-%d')
            except:
                pass  # Skip if conversion fails
    
    # Format amount columns with currency symbol
    amount_columns = [col for col in formatted_df.columns if 'amount' in col.lower() or col == 'amount']
    for col in amount_columns:
        if col in formatted_df.columns and formatted_df[col].dtype in [float, int]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"₹{x:,.2f}")
    
    # Format other numeric columns that might be currency
    numeric_currency_columns = ['total', 'sum', 'average', 'avg', 'balance']
    for col in formatted_df.columns:
        if any(term in col.lower() for term in numeric_currency_columns) and formatted_df[col].dtype in [float, int]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"₹{x:,.2f}")
    
    return formatted_df

def detect_visualization_type(df: pd.DataFrame, query_text: str) -> str:
    """
    Detect the best visualization type based on the query and data.
    
    Args:
        df: DataFrame with query results
        query_text: Original query text
        
    Returns:
        Visualization type ('bar', 'line', 'pie', 'table')
    """
    # Check if query suggests a time series (trend over time)
    time_keywords = ['month', 'year', 'quarter', 'weekly', 'daily', 'trend', 'over time']
    has_time_keywords = any(keyword in query_text.lower() for keyword in time_keywords)
    
    # Check if query suggests a comparison between categories
    comparison_keywords = ['compare', 'breakdown', 'distribution', 'split', 'proportion', 'percentage']
    has_comparison_keywords = any(keyword in query_text.lower() for keyword in comparison_keywords)
    
    # Look for date columns that could indicate a time series
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'month' in col.lower() or 'year' in col.lower()]
    has_date_column = len(date_columns) > 0
    
    # Look for category columns
    category_columns = [col for col in df.columns if 'category' in col.lower() or 'type' in col.lower()]
    has_category_column = len(category_columns) > 0
    
    # Check if there's a single numeric result (simple metric)
    is_single_metric = len(df) == 1 and df.select_dtypes(include=['number']).shape[1] == 1
    
    # Time series visualization (line chart)
    if has_time_keywords and has_date_column:
        return 'line'
    
    # Category comparison (bar chart)
    elif has_comparison_keywords and has_category_column:
        return 'bar'
    
    # Single row with multiple columns (pie chart)
    elif len(df) == 1 and df.select_dtypes(include=['number']).shape[1] > 1:
        return 'pie'
    
    # Multiple rows with 1-2 dimensions and 1 metric (bar chart)
    elif has_category_column and df.select_dtypes(include=['number']).shape[1] >= 1:
        return 'bar'
    
    # Multiple rows with date and metric (line chart)
    elif has_date_column and df.select_dtypes(include=['number']).shape[1] >= 1:
        return 'line'
    
    # Default to table view
    else:
        return 'table'

def display_transaction_visualization(df: pd.DataFrame, response: Dict[str, Any], viz_type: Optional[str] = None):
    """
    Display a visualization of transaction data based on the query result.
    
    Args:
        df: DataFrame with query results
        response: Complete response from the query handler
        viz_type: Optional override for visualization type
    """
    if df.empty:
        return
    
    # Determine visualization type if not provided
    if viz_type is None:
        viz_type = detect_visualization_type(df, response.get("query", ""))
    
    # Store the selected visualization type in session state
    st.session_state.visualization_type = viz_type
    
    # Visualization type selector
    viz_options = {
        'table': 'Table View',
        'bar': 'Bar Chart',
        'line': 'Line Chart',
        'pie': 'Pie Chart'
    }
    
    # Only show visualization options that make sense for this data
    available_viz_options = {}
    
    # Table is always available
    available_viz_options['table'] = 'Table View'
    
    # Bar chart is available if we have at least one categorical and one numeric column
    if (df.select_dtypes(include=['number']).shape[1] >= 1 and 
        df.select_dtypes(exclude=['number']).shape[1] >= 1):
        available_viz_options['bar'] = 'Bar Chart'
    
    # Line chart is available if we have date-like columns and numeric columns
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'month' in col.lower() or 'year' in col.lower()]
    if date_cols and df.select_dtypes(include=['number']).shape[1] >= 1:
        available_viz_options['line'] = 'Line Chart'
    
    # Pie chart is available for single rows with multiple numeric columns or 
    # categorical columns with one metric
    if (len(df) == 1 and df.select_dtypes(include=['number']).shape[1] > 1) or \
       (df.select_dtypes(include=['number']).shape[1] == 1 and 
        df.select_dtypes(exclude=['number']).shape[1] == 1 and
        len(df) <= 10):  # Limit to avoid overcrowded pie charts
        available_viz_options['pie'] = 'Pie Chart'
    
    # Select the visualization type
    selected_viz = st.radio(
        "Select Visualization Type:",
        options=list(available_viz_options.keys()),
        format_func=lambda x: available_viz_options[x],
        index=list(available_viz_options.keys()).index(viz_type) if viz_type in available_viz_options else 0,
        horizontal=True,
        key=f"viz_type_radio_{id(df)}" 
    )
    
    # Display the selected visualization
    if selected_viz == 'table':
        # Format the DataFrame for display
        display_df = format_transaction_dataframe(df)
        st.dataframe(display_df, use_container_width=True)
    
    elif selected_viz == 'bar':
        create_bar_chart(df)
    
    elif selected_viz == 'line':
        create_line_chart(df)
    
    elif selected_viz == 'pie':
        create_pie_chart(df)
    
    # Store the selected visualization type
    st.session_state.visualization_type = selected_viz

def create_bar_chart(df: pd.DataFrame):
    """Create a bar chart visualization for the data."""
    # Identify the best columns for the visualization
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    if not numeric_cols or not categorical_cols:
        st.warning("Cannot create bar chart: need both numeric and categorical columns")
        return
    
    # Try to make intelligent choices for x and y axes
    # Prefer 'category', 'type', 'name' columns for x-axis
    preferred_x_cols = [col for col in categorical_cols if col.lower() in ['category', 'type', 'name', 'description']]
    x_col = preferred_x_cols[0] if preferred_x_cols else categorical_cols[0]
    
    # Prefer 'amount', 'total', 'sum' columns for y-axis
    preferred_y_cols = [col for col in numeric_cols if col.lower() in ['amount', 'total', 'sum', 'count']]
    y_col = preferred_y_cols[0] if preferred_y_cols else numeric_cols[0]
    
    # If we have many rows, limit to top N for readability
    if len(df) > 10:
        df = df.sort_values(by=y_col, ascending=False).head(10)
        title = f"Top 10 by {y_col} (sorted)"
    else:
        title = f"{y_col} by {x_col}"
    
    # Create the bar chart
    fig = px.bar(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        labels={x_col: x_col.capitalize(), y_col: y_col.capitalize()},
        color=x_col if len(df[x_col].unique()) <= 10 else None,
        template="plotly_white"
    )
    
    # Format y-axis as currency if it seems to be a monetary value
    if any(term in y_col.lower() for term in ['amount', 'total', 'sum', 'balance']):
        fig.update_layout(yaxis_tickprefix='₹', yaxis_tickformat=',.0f')
    
    # Make sure x-axis labels are readable
    fig.update_layout(xaxis_tickangle=-45 if len(df) > 5 else 0)
    
    # Show the figure
    st.plotly_chart(fig, use_container_width=True, key=f"bar_chart_{id(df)}")

def create_line_chart(df: pd.DataFrame):
    """Create a line chart visualization for the data."""
    # Identify the best columns for the visualization
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Look for date/time/period columns
    date_patterns = ['date', 'month', 'year', 'quarter', 'period']
    date_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in date_patterns)]
    
    if not numeric_cols or not date_cols:
        st.warning("Cannot create line chart: need both numeric and date/period columns")
        return
    
    # Select the first date column for x-axis
    x_col = date_cols[0]
    
    # Try to convert x-axis to datetime if it's not already
    if df[x_col].dtype != 'datetime64[ns]':
        try:
            df[x_col] = pd.to_datetime(df[x_col])
        except:
            # If conversion fails, leave as is - might be a string period like "2023-01"
            pass
    
    # Prefer 'amount', 'total', 'sum' columns for y-axis
    preferred_y_cols = [col for col in numeric_cols if col.lower() in ['amount', 'total', 'sum', 'count']]
    y_col = preferred_y_cols[0] if preferred_y_cols else numeric_cols[0]
    
    # Create the line chart
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col,
        title=f"{y_col} over time",
        labels={x_col: x_col.capitalize(), y_col: y_col.capitalize()},
        markers=True,
        template="plotly_white"
    )
    
    # Format y-axis as currency if it seems to be a monetary value
    if any(term in y_col.lower() for term in ['amount', 'total', 'sum', 'balance']):
        fig.update_layout(yaxis_tickprefix='₹', yaxis_tickformat=',.0f')
    
    # Show the figure
    st.plotly_chart(fig, use_container_width=True, key=f"line_chart_{id(df)}")

def create_pie_chart(df: pd.DataFrame):
    """Create a pie chart visualization for the data."""
    # Case 1: One row with multiple numeric columns (use columns as categories)
    if len(df) == 1 and df.select_dtypes(include=['number']).shape[1] > 1:
        # Transpose the dataframe to get columns as rows
        numeric_df = df.select_dtypes(include=['number']).T.reset_index()
        numeric_df.columns = ['category', 'value']
        
        # Create pie chart
        fig = px.pie(
            numeric_df,
            values='value',
            names='category',
            title='Distribution by Category',
            template="plotly_white"
        )
    
    # Case 2: Multiple rows with one category and one metric
    elif df.select_dtypes(include=['number']).shape[1] == 1 and df.select_dtypes(exclude=['number']).shape[1] >= 1:
        numeric_col = df.select_dtypes(include=['number']).columns[0]
        
        # Choose the first non-numeric column as category
        category_col = df.select_dtypes(exclude=['number']).columns[0]
        
        # Create pie chart
        fig = px.pie(
            df,
            values=numeric_col,
            names=category_col,
            title=f'Distribution of {numeric_col} by {category_col}',
            template="plotly_white"
        )
    
    else:
        st.warning("Cannot create pie chart: need appropriate data structure")
        return
    
    # Update traces for better appearance
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    # Show the figure
    st.plotly_chart(fig, use_container_width=True, key=f"pie_chart_{id(df)}")