"""
Streamlit Web App for NLQ (Natural Language Query) System
A clean and attractive web interface for querying your sales data using natural language.
"""

import streamlit as st
import pandas as pd
import time
import os
from pathlib import Path
import sys

# Add the current directory to Python path to import from main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing NLQ system
from main import initialize_system, NLQSystem
from helper.types import QueryResult

# Page configuration
st.set_page_config(
    page_title="NLQ Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        color: #721c24; 
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #1e90ff;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .query-input {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .result-container {
        background-color: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'nlq_system' not in st.session_state:
    st.session_state.nlq_system = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

def initialize_nlq_system():
    """Initialize the NLQ system."""
    try:
        with st.spinner("🔄 Initializing NLQ System..."):
            system = initialize_system()
            st.session_state.nlq_system = system
            st.success("✅ NLQ System initialized successfully!")
            return system
    except Exception as e:
        st.error(f"❌ Failed to initialize NLQ System: {str(e)}")
        return None

def load_sales_data():
    """Load sales data into the system."""
    if not st.session_state.nlq_system:
        st.error("❌ NLQ System not initialized. Please initialize first.")
        return False
    
    try:
        with st.spinner("🔄 Loading sales data..."):
            # Check for data file
            data_files = [
                "./data/llm_dataset_v11.gz",
                "./data/llm_dataset_v11.gz",
                "./data/sales.csv",
                "./data/sales_data.csv"
            ]
            
            data_file = None
            for file_path in data_files:
                if os.path.exists(file_path):
                    data_file = file_path
                    break
            
            if not data_file:
                st.error("❌ No sales data file found. Please ensure one of these files exists:")
                for file_path in data_files:
                    st.write(f"  - {file_path}")
                return False
            
            # Load data
            load_result = st.session_state.nlq_system.load_data(data_file, "sales_data")
            st.session_state.data_loaded = True
            
            st.success(f"✅ Data loaded successfully! {load_result['total_rows']:,} rows in {load_result['duration']:.2f}s")
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")
        return False

def execute_query(query_text: str) -> QueryResult:
    """Execute a natural language query."""
    if not st.session_state.nlq_system:
        raise RuntimeError("NLQ System not initialized")
    
    if not st.session_state.data_loaded:
        raise RuntimeError("Data not loaded")
    
    return st.session_state.nlq_system.query(query_text, "sales_data")

def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame for better display in Streamlit."""
    formatted = df.copy()
    
    for col in formatted.columns:
        series = formatted[col]
        try:
            if pd.api.types.is_integer_dtype(series):
                formatted[col] = series.map(lambda x: f"{int(x):,}" if pd.notnull(x) else "")
            elif pd.api.types.is_float_dtype(series):
                formatted[col] = series.map(lambda x: f"{float(x):,.2f}" if pd.notnull(x) else "")
        except Exception:
            pass
    
    return formatted

def display_query_result(result: QueryResult, query_text: str):
    """Display query results in an attractive format."""
    st.markdown("### 📊 Query Results")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows Returned", f"{result.row_count:,}")
    
    with col2:
        st.metric("Execution Time", f"{result.execution_time:.3f}s")
    
    with col3:
        st.metric("Memory Usage", f"{result.memory_usage_mb:.1f}MB")
    
    with col4:
        cache_status = "🟢 Cached" if result.from_cache else "🟡 Fresh"
        st.metric("Cache Status", cache_status)
    
    # Show generated SQL
    with st.expander("🔍 Generated SQL Query", expanded=False):
        st.code(result.sql_query, language="sql")
    
    # Display results
    if result.row_count > 0:
        st.markdown("#### 📋 Data Results")
        
        # Format data for display
        display_df = format_dataframe_for_display(result.data)
        
        # Show data with pagination for large results
        if result.row_count > 100:
            st.info(f"📊 Large result set ({result.row_count:,} rows). Showing first 100 rows.")
            st.dataframe(display_df.head(100), use_container_width=True)
            
            # Add download button
            csv = result.data.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name=f"query_results_{int(time.time())}.csv",
                mime="text/csv"
            )
        else:
            st.dataframe(display_df, use_container_width=True)
            
            # Add download button for smaller results
            csv = result.data.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"query_results_{int(time.time())}.csv",
                mime="text/csv"
            )
    else:
        st.info("📭 No results found for this query.")

    # Textual summary (human language)
    if getattr(result, 'summary_text', ""):
        st.markdown("#### 📝 Summary")
        st.write(result.summary_text)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 NLQ Sales Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Ask questions about your sales data in plain English</p>', unsafe_allow_html=True)
    
    # Sidebar for system management
    with st.sidebar:
        st.markdown("## ⚙️ System Management")
        
        # Initialize system
        if st.button("🚀 Initialize NLQ System", use_container_width=True):
            initialize_nlq_system()
        
        # Load data
        if st.button("📁 Load Sales Data", use_container_width=True, disabled=not st.session_state.nlq_system):
            load_sales_data()
        
        # System status
        st.markdown("### 📊 System Status")
        if st.session_state.nlq_system:
            st.success("✅ NLQ System: Active")
        else:
            st.error("❌ NLQ System: Not Initialized")
        
        if st.session_state.data_loaded:
            st.success("✅ Data: Loaded")
        else:
            st.warning("⚠️ Data: Not Loaded")
        
        # Performance metrics
        if st.session_state.nlq_system and st.session_state.data_loaded:
            st.markdown("### 📈 Performance Metrics")
            try:
                report = st.session_state.nlq_system.get_performance_report()
                
                st.metric("Total Queries", report["query_metrics"]["total_queries"])
                st.metric("Cache Hit Rate", f"{report['cache_metrics']['hit_rate']:.1%}")
                st.metric("Avg Execution", f"{report['query_metrics']['average_execution_time']:.3f}s")
                
            except Exception as e:
                st.error(f"Error getting metrics: {e}")
        
        # Example queries
        st.markdown("### 💡 Example Queries")
        example_queries = [
            "What were the total sales in November 2024?",
            "Show the top 10 cities by sales in 2024",
            "What is the average sales per region?",
            "How many sales were made in Q1 2025?",
            "Which brands had the highest sales?",
            "What is the total revenue by month and year?"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{hash(query)}", use_container_width=True):
                st.session_state.example_query = query
                st.rerun()
    
    # Main content area
    if not st.session_state.nlq_system:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Welcome to NLQ Sales Analytics!</h3>
            <p>To get started:</p>
            <ol>
                <li>Click <strong>"Initialize NLQ System"</strong> in the sidebar</li>
                <li>Click <strong>"Load Sales Data"</strong> to load your dataset</li>
                <li>Start asking questions about your sales data!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
         
        return
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="info-box">
            <h3>📁 Data Loading Required</h3>
            <p>Please click <strong>"Load Sales Data"</strong> in the sidebar to load your sales dataset before running queries.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Query interface
    st.markdown("## 🔍 Ask a Question")
    
    # Query input (submit on Enter using form)
    with st.form(key="query_form", clear_on_submit=False):
        query_text = st.text_input(
            "Enter your question about the sales data:",
            placeholder="e.g., What were the total sales in Q3 2024?",
            key="query_input"
        )
        cols = st.columns([2, 1, 1])
        with cols[1]:
            execute_button = st.form_submit_button("🚀 Execute", use_container_width=True)
        with cols[2]:
            cancel_button = st.form_submit_button("🛑 Cancel", use_container_width=True)
    
    # Check if example query was selected
    if 'example_query' in st.session_state:
        query_text = st.session_state.example_query
        del st.session_state.example_query
        st.rerun()
    
    # Immediate cancel action
    if cancel_button:
        try:
            if st.session_state.nlq_system:
                canceled = st.session_state.nlq_system.cancel_running_query()
                if canceled:
                    st.warning("Query cancellation requested.")
                else:
                    st.info("No running query to cancel.")
        except Exception as e:
            st.error(f"Cancel failed: {e}")
    
    # Execute query
    if execute_button and query_text.strip():
        try:
            # Soft timeout watchdog: if > 10 minutes, show busy message
            start_time = time.time()
            with st.spinner("🤔 Processing your question..."):
                result = execute_query(query_text.strip())
                if time.time() - start_time > 600:
                    st.error("Server is too busy right now")
            
            # Add to query history
            st.session_state.query_history.append({
                'query': query_text.strip(),
                'timestamp': time.time(),
                'result': result
            })
            
            # Display results
            display_query_result(result, query_text.strip())
            
        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")
            st.markdown("""
            <div class="error-box">
                <h4>💡 Tips for better queries:</h4>
                <ul>
                    <li>Be specific about what you want to know</li>
                    <li>Mention time periods (e.g., "in 2024", "Q3")</li>
                    <li>Specify regions, brands, or other dimensions</li>
                    <li>Use natural language (e.g., "top 10", "average", "total")</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Query history
    if st.session_state.query_history:
        st.markdown("## 📚 Query History")
        
        for i, history_item in enumerate(reversed(st.session_state.query_history[-5:])):  # Show last 5 queries
            with st.expander(f"🔍 {history_item['query'][:50]}... ({time.strftime('%H:%M:%S', time.localtime(history_item['timestamp']))})", expanded=False):
                st.write(f"**Query:** {history_item['query']}")
                st.write(f"**Rows:** {history_item['result'].row_count:,}")
                st.write(f"**Time:** {history_item['result'].execution_time:.3f}s")
                
                if history_item['result'].row_count > 0:
                    st.dataframe(history_item['result'].data.head(5), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Powered by DuckDB and LLM-powered SQL generation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
