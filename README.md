# NLQ (Natural Language Query) System

A production-ready Natural Language Query system optimized for handling large datasets (1M+ rows) with memory efficiency and high performance.

## 🚀 Features

### Core Capabilities
- **Large Dataset Support**: Optimized for 1M+ row datasets with chunked processing
- **Memory Efficient**: Smart memory management with configurable limits
- **High Performance**: Connection pooling, query caching, and parallel processing
- **Production Ready**: Comprehensive logging, monitoring, and error handling

### Advanced Features
- **Intelligent Caching**: Multi-level caching for SQL queries and results
- **Auto-Configuration**: Automatic resource detection and optimization
- **Performance Monitoring**: Built-in benchmarking and profiling tools
- **Scalable Architecture**: Modular design with connection pooling

## 📋 Requirements

### System Requirements
- **Minimum**: 8GB RAM, 4-core CPU, 10GB storage
- **Recommended**: 16GB+ RAM, 8+ core CPU, SSD storage
- **Optional**: NVIDIA GPU with 8GB+ VRAM for acceleration

### Software Dependencies
```bash
pip install -r requirements.txt
```

## 🛠 Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd sqlcoder
   pip install -r requirements.txt
   ```

2. **Download the model**:
   ```bash
   # See models/README.md for detailed instructions
   python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='defog/llama-3-sqlcoder-8b-gguf', filename='llama-3-sqlcoder-8b.Q6_K.gguf', local_dir='models')"
   ```

3. **Prepare your data**:
   ```bash
   # Place your CSV file as sales.csv or modify the path in main.py
   cp your_data.csv sales.csv
   ```

## 🏃‍♂️ Quick Start

### Basic Usage
```python
from main import initialize_system, run_query

# Initialize the system
system = initialize_system()

# Load your data
system.load_data("sales.csv", "sales_data")

# Run queries
run_query("What were the total sales in Q3?")
run_query("Show me the top 5 regions by sales")
```

### Command Line
```bash
# Run the main application
python main.py

# Run benchmarks
python benchmark.py

# Create sample data for testing
python benchmark.py --create-sample
```

## ⚙️ Configuration

The system auto-configures based on your hardware, but you can customize settings in [`config.py`](config.py):

### Key Configuration Options
```python
# Database settings
chunk_size = 50000          # Rows per chunk for large files
memory_limit = "2GB"        # DuckDB memory limit
connection_pool_size = 5    # Database connection pool

# Model settings
n_ctx = 2048               # Context window size
n_threads = 6              # CPU threads for model
n_gpu_layers = 20          # GPU layers (if available)

# Caching settings
enable_query_cache = True   # Cache generated SQL
enable_result_cache = True  # Cache query results
cache_ttl = 3600           # Cache time-to-live (seconds)
```

### Environment Variables
```bash
export MODEL_PATH="models/your-model.gguf"
export DB_CHUNK_SIZE=25000
export LOG_LEVEL=DEBUG
export ENABLE_PROFILING=true
```

## 📊 Performance Optimization

### For Large Datasets (1M+ rows)
1. **Memory Management**:
   - Adjust `chunk_size` based on available RAM
   - Monitor memory usage with built-in tools
   - Use SSD storage for better I/O performance

2. **Query Optimization**:
   - Enable query and result caching
   - Use appropriate indexes (auto-created)
   - Leverage DuckDB's parallel processing

3. **Hardware Optimization**:
   - Use GPU acceleration if available
   - Increase CPU threads up to core count
   - Ensure sufficient RAM (16GB+ recommended)

### Performance Monitoring
```python
# Get performance report
report = system.get_performance_report()
print(json.dumps(report, indent=2))

# Run comprehensive benchmark
python benchmark.py
```

## 🏗 Architecture

### System Components
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NLQ Input     │───▶│  Query Cache     │───▶│  LLM Manager    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Result Cache   │◀───│ Database Manager │◀───│ SQL Generation │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Connection Pool  │
                       └──────────────────┘
```

### Key Classes
- **`NLQSystem`**: Main orchestrator
- **`DatabaseManager`**: Handles data loading and query execution
- **`LLMManager`**: Manages model inference and SQL generation
- **`QueryCache`**: Multi-level caching system
- **`MemoryMonitor`**: Resource monitoring and management

## 📈 Benchmarking

### Run Benchmarks
```bash
# Comprehensive benchmark suite
python benchmark.py

# Create sample data for testing
python benchmark.py --create-sample
```

### Benchmark Results
The system includes comprehensive benchmarking that measures:
- Query execution times
- Memory usage patterns
- Cache hit rates
- Scalability with different data sizes
- System resource utilization

Results are saved to `benchmarks/benchmark_report_TIMESTAMP.json`

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   ```python
   # Reduce chunk size in config.py
   config.database.chunk_size = 25000
   
   # Reduce model context
   config.model.n_ctx = 1024
   ```

2. **Slow Performance**:
   ```python
   # Increase threads (don't exceed CPU cores)
   config.model.n_threads = 8
   config.database.threads = 8
   
   # Enable GPU acceleration
   config.model.n_gpu_layers = 35
   ```

3. **Model Loading Issues**:
   - Verify model file exists and path is correct
   - Check available memory (model needs ~8GB)
   - Review logs in `logs/nlq_system.log`

### Performance Tuning
- Monitor system resources during operation
- Adjust configuration based on your hardware
- Use the built-in benchmarking tools to measure improvements
- Check the performance report for bottlenecks

## 📝 Logging

Logs are written to `logs/nlq_system.log` with configurable levels:
- **INFO**: General operation information
- **DEBUG**: Detailed execution information
- **WARNING**: Performance warnings and issues
- **ERROR**: Error conditions and failures

## 🔒 Security Considerations

- Model files are excluded from version control (see `.gitignore`)
- SQL injection protection through query validation
- Memory usage monitoring to prevent DoS
- Configurable resource limits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the benchmark suite
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **DuckDB**: High-performance analytical database
- **llama.cpp**: Efficient LLM inference
- **Pandas**: Data manipulation and analysis
- **SQLCoder**: Specialized SQL generation model

## 📚 Additional Resources

- [Models Documentation](models/README.md)
- [Configuration Guide](config.py)
- [Benchmarking Guide](benchmark.py)
- [Performance Optimization Tips](#performance-optimization)

---

**Built for production workloads with 1M+ row datasets** 🚀
