# NLQ Web Application - Chat CPG

A web-based Natural Language Query (NLQ) system that allows users to ask questions about their data in plain English and receive SQL-generated results with AI-powered summaries.

## Features

- **Natural Language Interface**: Ask questions about your data in plain English
- **SQL Generation**: Uses SQL Coder model to convert NLQ to SQL
- **AI Summarization**: Mistral model provides human-readable summaries of results
- **Interactive Chat**: Real-time chat interface with query history
- **Data Management**: Load datasets and manage system initialization
- **History Tracking**: Persistent storage of queries and results
- **No Currency Symbols**: Summaries avoid currency units and use human-readable number formats

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your dataset file in the `data/` directory:
- Supported formats: CSV, GZ compressed CSV
- Default path: `data/llm_dataset_v11.gz`
- Table name: `sales_data` (configurable)

### 3. Run the Application

```bash
python run_web.py
```

Or directly with Flask:
```bash
python app.py
```

### 4. Access the Web Interface

Open your browser and go to: `http://localhost:5000`

## Usage Guide

### Step 1: Initialize System
1. Click the **"Initialize System"** button in the sidebar
2. Wait for the system to load the NLQ models
3. Status indicator will turn green when ready

### Step 2: Load Dataset
1. Click the **"Load Dataset"** button
2. The system will load your data file
3. Status will show "Dataset Loaded" when complete

### Step 3: Ask Questions
1. Type your question in the chat input
2. Press Enter or click Send
3. View the generated SQL, results, and AI summary

### Example Queries

- "Show top 5 territories by sales in December"
- "What were the total sales in Q3 2024?"
- "Which brands had the highest growth last month?"
- "List territories that missed their targets"
- "Show productivity rates by region"

## System Architecture

```
User Query (NLQ) → SQL Coder Model → SQL Generation → Database Execution → Mistral Summarizer → Human Summary
```

### Components

- **Frontend**: HTML/CSS/JavaScript chat interface
- **Backend**: Flask web server with REST API
- **NLQ System**: SQL generation and query execution
- **Models**: SQL Coder (SQL generation) + Mistral (summarization)
- **Database**: DuckDB for fast data processing
- **History**: Session-based query storage

## API Endpoints

- `POST /api/initialize` - Initialize the NLQ system
- `POST /api/load_dataset` - Load a dataset
- `POST /api/query` - Process a natural language query
- `GET /api/history` - Get query history
- `GET /api/history/<id>` - Get specific history item
- `POST /api/clear_history` - Clear query history

## Configuration

### Model Paths
Update `config.py` to point to your model files:
```python
model_path: str = "models/llama-3-sqlcoder-8b.Q6_K.gguf"
summarizer_model_path: str = "models/Mistral-7B-Instruct-v0.1.Q6_K.gguf"
```

### Dataset Path
Modify the default dataset path in the web interface or API calls:
```javascript
dataset_path: 'data/your_dataset.csv'
```

## Troubleshooting

### Common Issues

1. **"Model not found"**
   - Ensure model files exist in the specified paths
   - Check file permissions

2. **"Dataset not found"**
   - Verify dataset file exists in `data/` directory
   - Check file path in the load dataset request

3. **"System initialization failed"**
   - Check if all required Python packages are installed
   - Verify model files are accessible

4. **"Query failed"**
   - Ensure system is initialized and dataset is loaded
   - Check if the question is clear and specific
   - Review error messages for specific issues

### Performance Tips

- Use specific time periods (e.g., "December 2024" instead of "last month")
- Be specific about dimensions (e.g., "territories" instead of "areas")
- For large datasets, consider adding LIMIT clauses in your questions
- Use the history feature to avoid repeating similar queries

## Development

### Project Structure
```
├── app.py                 # Flask web application
├── run_web.py            # Startup script
├── templates/
│   └── view.html        # Web interface template
├── helper/               # NLQ system modules
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
└── README_WEB.md       # This file
```

### Adding New Features

1. **New API Endpoints**: Add routes in `app.py`
2. **UI Enhancements**: Modify `templates/view.html`
3. **NLQ Improvements**: Update `helper/nlq_system.py`
4. **Configuration**: Modify `config.py`

### Testing

```bash
# Run the application
python run_web.py

# Test API endpoints
curl -X POST http://localhost:5000/api/initialize
curl -X POST http://localhost:5000/api/load_dataset
curl -X POST http://localhost:5000/api/query -H "Content-Type: application/json" -d '{"query":"Show top 5 territories"}'
```

## Security Notes

- Change the secret key in `app.py` for production use
- Consider adding authentication for production deployments
- Validate and sanitize all user inputs
- Use HTTPS in production environments

## License

This project is part of the SQL Coder NLQ system. Please refer to the main project license for usage terms.
