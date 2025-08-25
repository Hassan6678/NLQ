"""
Streamlit Web App for NLQ (Natural Language Query) System
A clean and attractive web interface for querying your sales data using natural language.
"""

from flask import Flask, render_template, request, jsonify, session
import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import glob

from config import config
from helper.nlq_system import NLQSystem

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Configure logging
def setup_logging():
    """Setup file-based logging with rotation."""
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'app_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    # Clean up old log files
    cleanup_old_logs(logs_dir)
    
    return logging.getLogger(__name__)

def cleanup_old_logs(logs_dir: Path, max_files: int = 12):
    """Delete old log files if the number exceeds the limit."""
    try:
        # Get all log files sorted by modification time (oldest first)
        log_files = sorted(
            logs_dir.glob('app_*.log'),
            key=lambda x: x.stat().st_mtime
        )
        
        # Remove old files if we exceed the limit
        if len(log_files) > max_files:
            files_to_remove = log_files[:-max_files]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    print(f"Removed old log file: {file_path}")
                except Exception as e:
                    print(f"Failed to remove old log file {file_path}: {e}")
    except Exception as e:
        print(f"Error during log cleanup: {e}")

# Setup logging
logger = setup_logging()

# Global NLQ system instance
nlq_system: Optional[NLQSystem] = None

def get_nlq_system() -> NLQSystem:
    """Get or initialize the NLQ system."""
    global nlq_system
    if nlq_system is None:
        nlq_system = NLQSystem()
    return nlq_system

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('view.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    logger.info("Health check requested")
    try:
        system = get_nlq_system()
        loaded_tables = list(system.loaded_tables.keys())
        logger.info(f"Health check successful - loaded tables: {loaded_tables}")
        return jsonify({
            'success': True,
            'status': 'healthy',
            'system_initialized': True,
            'loaded_tables': loaded_tables,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize the NLQ system."""
    logger.info("System initialization requested")
    try:
        system = get_nlq_system()
        logger.info("System initialized successfully")
        return jsonify({
            'success': True,
            'message': 'System initialized successfully'
        })
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Initialization failed: {str(e)}'
        }), 500

@app.route('/api/load_dataset', methods=['POST'])
def load_dataset():
    """Load a dataset into the system."""
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path', 'data/llm_dataset_v11.gz')
        table_name = data.get('table_name', 'sales_data')
        
        logger.info(f"Dataset load requested: {dataset_path} -> {table_name}")
        
        # Expand and resolve path
        dataset_path = str(Path(dataset_path).expanduser().resolve())
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found: {dataset_path}")
            return jsonify({
                'success': False,
                'message': f'Dataset not found: {dataset_path}'
            }), 404
        
        system = get_nlq_system()
        # If table already present skip
        if system.fast_attach_existing(table_name):
            logger.info("Using existing persisted table – skipping raw load")
            result = system.loaded_tables[table_name]
        else:
            result = system.load_data(dataset_path, table_name)
        
        logger.info(f"Dataset loaded successfully: {result['total_rows']:,} rows into {table_name}")
        return jsonify({
            'success': True,
            'message': f'Loaded {result["total_rows"]:,} rows into {table_name}',
            'data': result
        })
    except Exception as e:
        logger.error(f"Dataset load failed: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Failed to load dataset: {str(e)}'
        }), 500

@app.route('/api/refresh_table', methods=['POST'])
def refresh_table():
    """Force reload (drop and re-load) the table from dataset path."""
    try:
        data = request.get_json() or {}
        dataset_path = data.get('dataset_path', 'data/llm_dataset_v11.gz')
        table_name = data.get('table_name', 'sales_data')
        system = get_nlq_system()
        system.reset_table(table_name)
        result = system.load_data(str(Path(dataset_path).expanduser().resolve()), table_name)
        return jsonify({'success': True, 'message': 'Table refreshed', 'data': result})
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        return jsonify({'success': False, 'message': f'Refresh failed: {e}'}), 500

@app.route('/api/delete_db', methods=['POST'])
def delete_db():
    """Delete entire DuckDB file for a clean rebuild."""
    try:
        system = get_nlq_system()
        ok = system.delete_database()
        return jsonify({'success': ok, 'message': 'Database deleted' if ok else 'Delete failed'})
    except Exception as e:
        logger.error(f"Delete DB failed: {e}")
        return jsonify({'success': False, 'message': f'Delete failed: {e}'}), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a natural language query and return results."""
    try:
        data = request.get_json(silent=True) or {}
        nlq = data.get('query', '').strip()
        
        logger.info(f"Query received: {nlq}")
        
        if not nlq:
            logger.warning("Empty query received")
            return jsonify({
                'success': False,
                'message': 'Query cannot be empty'
            }), 400
        
        system = get_nlq_system()
        
        # Process the query (no timeout)
        start_time = time.time()
        try:
            result = system.query(nlq)
            logger.info(f"Query processed successfully: {result.row_count} rows returned")
        except Exception as e:
            # Return structured error to the UI
            error_msg = f'{type(e).__name__}: {str(e)}'
            logger.error(f"Query processing failed: {error_msg}")
            return jsonify({
                'success': False,
                'message': error_msg
            }), 500
            
        total_time = time.time() - start_time
        
        # Format the response
        response = {
            'success': True,
            'query': nlq,
            'sql': result.sql_query,
            'summary': result.summary_text,
            'row_count': result.row_count,
            'execution_time': result.execution_time,
            'total_time': total_time,
            'data': result.data.to_dict('records') if result.row_count > 0 else [],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Store in session history
        if 'query_history' not in session:
            session['query_history'] = []
        
        session['query_history'].append({
            'id': len(session['query_history']) + 1,
            'query': nlq,
            'summary': result.summary_text,
            'timestamp': response['timestamp'],
            'row_count': result.row_count
        })
        
        # Keep only last 50 queries
        if len(session['query_history']) > 50:
            session['query_history'] = session['query_history'][-50:]
        
        logger.info(f"Query completed successfully in {total_time:.3f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Unexpected error: {str(e)}'
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get query history."""
    logger.info("History requested")
    history = session.get('query_history', [])
    return jsonify({
        'success': True,
        'history': history
    })

@app.route('/api/history/<int:query_id>', methods=['GET'])
def get_history_item(query_id):
    """Get a specific history item."""
    history = session.get('query_history', [])
    item = next((h for h in history if h['id'] == query_id), None)
    
    if item is None:
        return jsonify({
            'success': False,
            'message': 'History item not found'
        }), 404
    
    return jsonify({
        'success': True,
        'item': item
    })

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear query history."""
    logger.info("History clear requested")
    session['query_history'] = []
    return jsonify({
        'success': True,
        'message': 'History cleared'
    })

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=5000)
