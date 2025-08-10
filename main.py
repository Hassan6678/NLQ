"""
Production-ready NLQ (Natural Language Query) system for large datasets.
Optimized for handling 1M+ rows with memory efficiency and performance.
"""

import os
import sys
import time
import logging
import hashlib
import pickle
import threading
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import json
import argparse
import re

import pandas as pd
import duckdb
import psutil
from llama_cpp import Llama

from config import config


# Setup logging
def setup_logging():
    """Configure logging with rotation and performance tracking."""
    os.makedirs(os.path.dirname(config.logging.file_path), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.FileHandler(config.logging.file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


logger = setup_logging()


@dataclass
class QueryResult:
    """Container for query results with metadata."""
    data: pd.DataFrame
    sql_query: str
    execution_time: float
    row_count: int
    from_cache: bool = False
    memory_usage_mb: float = 0.0


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    query_count: int = 0
    total_execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_peak_mb: float = 0.0
    errors: int = 0


class MemoryMonitor:
    """Monitor memory usage and prevent OOM errors."""
    
    def __init__(self):
        self.peak_usage = 0.0
        self.current_usage = 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        usage_mb = process.memory_info().rss / 1024 / 1024
        self.current_usage = usage_mb
        self.peak_usage = max(self.peak_usage, usage_mb)
        return usage_mb
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        memory_info = config.get_memory_info()
        current_usage_gb = self.get_memory_usage() / 1024
        return current_usage_gb < memory_info["max_allowed_gb"]
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager to monitor memory during operations."""
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            duration = time.time() - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(f"{operation_name} - Duration: {duration:.2f}s, "
                       f"Memory delta: {memory_delta:.1f}MB, "
                       f"Peak: {self.peak_usage:.1f}MB")


class QueryCache:
    """Intelligent caching system for queries and results."""
    
    def __init__(self):
        self.query_cache = {}
        self.result_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        os.makedirs(config.cache.cache_dir, exist_ok=True)
    
    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query caching."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get_cached_sql(self, nlq: str) -> Optional[str]:
        """Get cached SQL for natural language query."""
        if not config.cache.enable_query_cache:
            return None
        
        query_hash = self._get_query_hash(nlq)
        if query_hash in self.query_cache:
            self.cache_stats["hits"] += 1
            logger.debug(f"SQL cache hit for query: {nlq[:50]}...")
            return self.query_cache[query_hash]
        
        self.cache_stats["misses"] += 1
        return None
    
    def cache_sql(self, nlq: str, sql: str):
        """Cache generated SQL."""
        if not config.cache.enable_query_cache:
            return
        
        query_hash = self._get_query_hash(nlq)
        self.query_cache[query_hash] = sql
        
        # Limit cache size
        if len(self.query_cache) > config.cache.max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
    
    def get_cached_result(self, sql: str) -> Optional[QueryResult]:
        """Get cached query result."""
        if not config.cache.enable_result_cache:
            return None
        
        result_hash = self._get_query_hash(sql)
        cache_file = os.path.join(config.cache.cache_dir, f"{result_hash}.pkl")
        
        if os.path.exists(cache_file):
            try:
                # Check if cache is still valid (TTL)
                cache_age = time.time() - os.path.getmtime(cache_file)
                if cache_age < config.cache.cache_ttl:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                        result.from_cache = True
                        self.cache_stats["hits"] += 1
                        logger.debug(f"Result cache hit for SQL: {sql[:50]}...")
                        return result
                else:
                    # Remove expired cache
                    os.remove(cache_file)
            except Exception as e:
                logger.warning(f"Error loading cache file {cache_file}: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    def cache_result(self, sql: str, result: QueryResult):
        """Cache query result."""
        if not config.cache.enable_result_cache:
            return
        
        result_hash = self._get_query_hash(sql)
        cache_file = os.path.join(config.cache.cache_dir, f"{result_hash}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Error caching result: {e}")


class DatabaseManager:
    """Optimized database manager with connection pooling and chunked processing."""
    
    def __init__(self):
        self.connections = []
        self.connection_lock = threading.Lock()
        self.table_schemas = {}
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize connection pool."""
        for i in range(config.database.connection_pool_size):
            # Use a shared on-disk DuckDB database so all pooled connections
            # see the same tables loaded during the session.
            conn = duckdb.connect(config.database.db_path)
            # Optimize DuckDB settings
            conn.execute(f"SET memory_limit='{config.database.memory_limit}'")
            conn.execute(f"SET threads={config.database.threads}")
            conn.execute(f"SET enable_progress_bar=false")
            if config.database.enable_parallel:
                conn.execute("PRAGMA threads=4")
                # conn.execute("SET enable_parallel=true")
            self.connections.append(conn)
        
        logger.info(f"Initialized {len(self.connections)} database connections")
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool."""
        with self.connection_lock:
            if self.connections:
                conn = self.connections.pop()
            else:
                # Create new connection if pool is empty
                conn = duckdb.connect(config.database.db_path)
                logger.warning("Connection pool exhausted, creating new connection")
        
        try:
            yield conn
        finally:
            with self.connection_lock:
                self.connections.append(conn)
    
    def load_csv_chunked(self, file_path: str, table_name: str) -> Dict[str, Any]:
        """Load large CSV file using chunked processing."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        logger.info(f"Loading CSV file: {file_path}")
        start_time = time.time()
        total_rows = 0
        
        # Get file size for progress tracking
        file_size_mb = os.path.getsize(file_path) / 1024 / 1024
        logger.info(f"File size: {file_size_mb:.1f}MB")
        
        with self.get_connection() as conn:
            # Drop table if exists
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            first_chunk = True
            chunk_count = 0
            
            try:
                # Read CSV in chunks
                for chunk in pd.read_csv(file_path, chunksize=config.database.chunk_size, low_memory=False):
                    # Heuristically coerce dtypes so numeric/boolean columns are not treated as text
                    chunk = self._coerce_chunk_dtypes(chunk)
                    chunk_count += 1
                    chunk_rows = len(chunk)
                    total_rows += chunk_rows
                    
                    if first_chunk:
                        # Create table with first chunk using a temporary view name
                        temp_initial = f"temp_initial_chunk_{int(time.time())}"
                        conn.register(temp_initial, chunk)
                        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {temp_initial}")
                        conn.execute(f"DROP VIEW IF EXISTS {temp_initial}")
                        
                        # Store schema information (after coercion)
                        schema_info = []
                        for col, dtype in chunk.dtypes.items():
                            sql_type = self._pandas_to_sql_type(dtype)
                            schema_info.append(f"{self._quote_identifier(col)} {sql_type}")
                        
                        self.table_schemas[table_name] = {
                            "columns": list(chunk.columns),
                            "schema": ", ".join(schema_info),
                            "sample_data": chunk.head(3).to_dict('records'),
                            "table_name": table_name
                        }
                        
                        first_chunk = False
                        logger.info(f"Created table {table_name} with {chunk_rows} rows")
                    else:
                        # Insert subsequent chunks
                        temp_table = f"temp_chunk_{chunk_count}"
                        conn.register(temp_table, chunk)
                        conn.execute(f"INSERT INTO {table_name} SELECT * FROM {temp_table}")
                        conn.execute(f"DROP VIEW IF EXISTS {temp_table}")
                    
                    # Log progress
                    if chunk_count % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = total_rows / elapsed if elapsed > 0 else 0
                        logger.info(f"Processed {chunk_count} chunks, {total_rows:,} rows "
                                   f"({rate:.0f} rows/sec)")
                        
                        # Check memory usage
                        if not memory_monitor.check_memory_limit():
                            logger.warning("Memory usage approaching limit, consider reducing chunk size")
                
                # Create indexes for better query performance
                self._create_indexes(conn, table_name)
                
                duration = time.time() - start_time
                logger.info(f"Successfully loaded {total_rows:,} rows in {duration:.2f}s "
                           f"({total_rows/duration:.0f} rows/sec)")
                
                return {
                    "table_name": table_name,
                    "total_rows": total_rows,
                    "duration": duration,
                    "chunks_processed": chunk_count,
                    "schema": self.table_schemas[table_name]
                }
                
            except Exception as e:
                logger.error(f"Error loading CSV: {e}")
                # Cleanup on error
                try:
                    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                except:
                    pass
                raise
    
    def _pandas_to_sql_type(self, dtype) -> str:
        """Convert pandas dtype to SQL type."""
        if pd.api.types.is_integer_dtype(dtype):
            return "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            # Prefer DECIMAL for money-like columns; DOUBLE otherwise.
            return "DOUBLE"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "TIMESTAMP"
        elif pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        else:
            return "VARCHAR"
    
    def _create_indexes(self, conn, table_name: str):
        """Create indexes for common query patterns."""
        try:
            # Get table info to identify potential index columns
            result = conn.execute(f"DESCRIBE {table_name}").fetchall()
            columns = [row[0] for row in result]
            
            # Create indexes on common column patterns
            index_patterns = [
                'id', 'date', 'time', 'region', 'category', 'type',
                'month', 'year', 'city', 'brand', 'sku', 'customer',
                'territory', 'route', 'area', 'distributor'
            ]
            
            for col in columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in index_patterns):
                    try:
                        conn.execute(f"CREATE INDEX idx_{table_name}_{col} ON {table_name}({col})")
                        logger.debug(f"Created index on {table_name}.{col}")
                    except Exception as e:
                        logger.debug(f"Could not create index on {col}: {e}")
                        
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def _coerce_chunk_dtypes(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Attempt to coerce object-typed columns to numeric or boolean where appropriate.
        This prevents saving numeric columns (e.g., "primary sales") as VARCHAR.
        """
        try:
            coerced = chunk.copy()
            # Pre-known boolean-like columns
            boolean_candidates = {"productivity", "stockout", "assortment"}
            for col in coerced.columns:
                series = coerced[col]
                # Force month/year to integers if possible
                if col.lower() in {"month", "year"}:
                    coerced[col] = pd.to_numeric(series, errors="coerce").astype("Int64")
                    continue
                if pd.api.types.is_bool_dtype(series):
                    continue
                if pd.api.types.is_numeric_dtype(series):
                    continue
                if pd.api.types.is_object_dtype(series):
                    # Try boolean coercion first
                    if col.lower() in boolean_candidates:
                        str_vals = series.astype(str).str.strip().str.lower()
                        true_set = {"true", "1", "yes"}
                        false_set = {"false", "0", "no"}
                        if str_vals.isin(true_set | false_set).mean() > 0.9:
                            coerced[col] = str_vals.map(lambda v: True if v in true_set else (False if v in false_set else pd.NA)).astype("boolean")
                            continue
                    # Try numeric coercion with cleanup for commas and currency symbols
                    cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
                    numeric = pd.to_numeric(cleaned, errors="coerce")
                    non_null_ratio = numeric.notna().mean()
                    if non_null_ratio > 0.9:  # mostly numeric
                        # Choose integer if no fractional part, else float
                        if (numeric.dropna() % 1 == 0).all():
                            coerced[col] = numeric.astype("Int64")
                        else:
                            coerced[col] = numeric.astype("float64")
                        continue
            return coerced
        except Exception:
            return chunk
    
    def execute_query(self, sql: str) -> QueryResult:
        """Execute SQL query with performance monitoring."""
        start_time = time.time()
        
        with self.get_connection() as conn:
            try:
                result_df = conn.execute(sql).fetchdf()
                execution_time = time.time() - start_time
                memory_usage = memory_monitor.get_memory_usage()
                
                query_result = QueryResult(
                    data=result_df,
                    sql_query=sql,
                    execution_time=execution_time,
                    row_count=len(result_df),
                    memory_usage_mb=memory_usage
                )
                
                logger.info(f"Query executed in {execution_time:.3f}s, "
                           f"returned {len(result_df):,} rows")
                
                return query_result
                
            except Exception as e:
                logger.error(f"SQL execution error: {e}")
                logger.error(f"SQL: {sql}")
                raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive table information."""
        if table_name not in self.table_schemas:
            return {}
        
        with self.get_connection() as conn:
            try:
                # Get row count
                row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                
                # Get column statistics
                columns = self.table_schemas[table_name]["columns"]
                stats = {}
                
                for col in columns[:5]:  # Limit to first 5 columns for performance
                    try:
                        result = conn.execute(f"""
                            SELECT 
                                COUNT(DISTINCT {col}) as unique_count,
                                COUNT({col}) as non_null_count
                            FROM {table_name}
                        """).fetchone()
                        
                        stats[col] = {
                            "unique_count": result[0],
                            "non_null_count": result[1]
                        }
                    except:
                        pass
                
                return {
                    **self.table_schemas[table_name],
                    "row_count": row_count,
                    "column_stats": stats
                }
                
            except Exception as e:
                logger.error(f"Error getting table info: {e}")
                return self.table_schemas[table_name]

    def _quote_identifier(self, name: str) -> str:
        """Quote identifier if it contains spaces or non-word characters.
        DuckDB uses double quotes for quoted identifiers.
        """
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            return name
        return '"' + name.replace('"', '""') + '"'


class LLMManager:
    """Optimized LLM manager with context management and prompt engineering."""
    
    def __init__(self):
        self.llm = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model with optimized settings."""
        try:
            logger.info(f"Loading model: {config.model.model_path}")
            start_time = time.time()
            
            self.llm = Llama(
                model_path=config.model.model_path,
                n_ctx=config.model.n_ctx,
                n_threads=config.model.n_threads,
                n_gpu_layers=config.model.n_gpu_layers,
                verbose=config.model.verbose
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def build_enhanced_prompt(self, nlq: str, table_info: Dict[str, Any]) -> str:
        """Build enhanced prompt with table context and examples."""
        schema = table_info.get("schema", "")
        sample_data = table_info.get("sample_data", [])
        row_count = table_info.get("row_count", 0)
        table_name_for_prompt = table_info.get("table_name", "sales_data")
        
        # Build sample data context
        sample_context = ""
        if sample_data:
            sample_context = "\n### Sample data (first 3 rows):\n"
            for i, row in enumerate(sample_data[:3], 1):
                sample_context += f"Row {i}: {row}\n"
        
        # Build column statistics context
        stats_context = ""
        column_stats = table_info.get("column_stats", {})
        if column_stats:
            stats_context = "\n### Column information:\n"
            for col, stats in column_stats.items():
                unique_pct = (stats["unique_count"] / row_count * 100) if row_count > 0 else 0
                stats_context += f"- {col}: {stats['unique_count']:,} unique values ({unique_pct:.1f}%)\n"
        
        prompt = f"""### You are an expert SQL generator for DuckDB.
### Given the following table schema and context:

# Table: {table_name_for_prompt}
# Schema: {schema}
# Total rows: {row_count:,}
{sample_context}
{stats_context}

### Important guidelines:
- Use DuckDB SQL syntax
 - The ONLY available table is named "{table_name_for_prompt}". You must use exactly this table name in all FROM and JOIN clauses. Do not invent any other table names.
- For large datasets, consider using LIMIT for exploratory queries
- Use appropriate aggregations for summary queries
- Handle NULL values appropriately
- Use proper date/time functions if applicable
 - Columns may include names with spaces (e.g., "primary sales"). When referencing such columns, use double quotes, e.g., "primary sales".
 - Months and years are integers. If the question mentions a quarter (Q1..Q4), map to months: Q1={1,2,3}, Q2={4,5,6}, Q3={7,8,9}, Q4={10,11,12}.
 - Booleans are stored as TRUE/FALSE. For boolean filters, use syntax like stockout = TRUE or productivity = FALSE.

### Write a SQL query to answer the question:
# {nlq}

### SQL:
SELECT"""
        
        return prompt
    
    def generate_sql(self, prompt: str) -> Optional[str]:
        """Generate SQL with enhanced error handling and validation."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            start_time = time.time()
            
            output = self.llm(
                prompt,
                temperature=config.model.temperature,
                max_tokens=config.model.max_tokens,
                stop=[
                    "###",
                    "\n\n",
                    "assistant",
                    "Assistant",
                    "ASSISTANT",
                    "User:",
                    "USER:",
                    "```",
                    "<|assistant|>",
                    "</s>"
                ]
            )
            
            generation_time = time.time() - start_time
            text = output["choices"][0]["text"]
            
            logger.debug(f"LLM generation time: {generation_time:.3f}s")
            
            # Enhanced SQL extraction and validation
            sql = self._extract_and_validate_sql(text)
            
            if sql:
                logger.debug(f"Generated SQL: {sql}")
                return sql
            else:
                logger.warning("Could not extract valid SQL from model output")
                logger.debug(f"Raw output: {text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return None
    
    def _extract_and_validate_sql(self, text: str) -> Optional[str]:
        """Extract and validate SQL from model output."""
        try:
            # Clean up the text
            text = text.strip()
            
            # If SELECT is not in the text, prepend it
            if "SELECT" not in text.upper():
                if text:
                    text = "SELECT " + text
                else:
                    return None
            
            # Extract SQL statement
            text_upper = text.upper()
            if "SELECT" in text_upper:
                sql_start = text_upper.find("SELECT")
                sql_text = text[sql_start:]

                # Find the earliest end delimiter (case-insensitive)
                sql_end = len(sql_text)
                lower_sql_text = sql_text.lower()
                delimiters = [
                    ";",
                    "\n\n",
                    "###",
                    "assistant",
                    "user:",
                    "```",
                    "<|assistant|>",
                    "</s>"
                ]
                for delimiter in delimiters:
                    pos = lower_sql_text.find(delimiter)
                    if pos != -1:
                        sql_end = min(sql_end, pos)

                sql = sql_text[:sql_end].strip()
                
                # Add semicolon if not present
                if not sql.endswith(";"):
                    sql += ";"
                
                # Basic validation
                if self._validate_sql_syntax(sql):
                    return sql
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting SQL: {e}")
            return None
    
    def _validate_sql_syntax(self, sql: str) -> bool:
        """Basic SQL syntax validation."""
        sql_upper = sql.upper()
        
        # Check for required SELECT
        if not sql_upper.startswith("SELECT"):
            return False
        
        # Check for dangerous operations (basic safety)
        # Allow CREATE VIEW/TABLE when we are the ones creating during load,
        # but prevent it in generated query. Keep validation conservative.
        dangerous_keywords = [" DROP ", " DELETE ", " TRUNCATE ", " ALTER ", " CREATE "]
        for keyword in dangerous_keywords:
            if keyword in f" {sql_upper} ":
                logger.warning(f"Potentially dangerous SQL keyword detected: {keyword}")
                return False
        
        # Check for balanced parentheses
        if sql.count("(") != sql.count(")"):
            return False
        
        return True


class NLQSystem:
    """Main NLQ system orchestrating all components."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.llm_manager = LLMManager()
        self.cache = QueryCache()
        self.metrics = PerformanceMetrics()
        self.loaded_tables = {}
        
        logger.info("NLQ System initialized successfully")
    
    def load_data(self, file_path: str, table_name: str = "sales_data") -> Dict[str, Any]:
        """Load data file with optimized processing."""
        with memory_monitor.monitor_operation(f"Loading {file_path}"):
            result = self.db_manager.load_csv_chunked(file_path, table_name)
            self.loaded_tables[table_name] = result
            return result
    
    def query(self, nlq: str, table_name: str = "sales_data") -> QueryResult:
        """Execute natural language query with full optimization pipeline."""
        start_time = time.time()
        
        try:
            # Check if table exists
            if table_name not in self.loaded_tables:
                raise ValueError(f"Table {table_name} not loaded. Load data first.")
            
            # Check SQL cache first
            cached_sql = self.cache.get_cached_sql(nlq)
            
            if cached_sql:
                sql = cached_sql
                logger.info("Using cached SQL")
            else:
                # Generate SQL using LLM
                table_info = self.db_manager.get_table_info(table_name)
                prompt = self.llm_manager.build_enhanced_prompt(nlq, table_info)
                sql = self.llm_manager.generate_sql(prompt)
                
                if not sql:
                    raise ValueError("Could not generate valid SQL from query")
                
                # Cache the generated SQL
                self.cache.cache_sql(nlq, sql)
            
            # Normalize/ensure the correct table name is used in the SQL
            sql = self._normalize_sql_table_name(sql, table_name)
            # Quote any column identifiers that require quoting (e.g., contain spaces)
            sql = self._quote_columns_with_spaces_in_sql(sql, table_name)
            # Enforce semantic column preferences (e.g., prefer "primary sales" when requested)
            sql = self._enforce_semantic_column_preference(nlq, sql, table_name)
            
            # Check result cache
            cached_result = self.cache.get_cached_result(sql)
            if cached_result:
                logger.info("Using cached result")
                self.metrics.cache_hits += 1
                return cached_result
            
            # Execute query
            with memory_monitor.monitor_operation("Query execution"):
                result = self.db_manager.execute_query(sql)
            
            # Cache the result
            self.cache.cache_result(sql, result)
            
            # Update metrics
            self.metrics.query_count += 1
            self.metrics.total_execution_time += result.execution_time
            self.metrics.cache_misses += 1
            self.metrics.memory_peak_mb = max(self.metrics.memory_peak_mb, result.memory_usage_mb)
            
            total_time = time.time() - start_time
            logger.info(f"Query completed in {total_time:.3f}s total")
            
            return result
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"Query failed: {e}")
            raise

    def _normalize_sql_table_name(self, sql: str, expected_table: str) -> str:
        """Ensure generated SQL references the expected table name.
        Rewrites FROM/JOIN targets that don't match `expected_table`.
        """
        try:
            # Also strip stray chat tokens like trailing 'assistant' that may have
            # slipped past stop tokens and delimiters.
            sql = re.sub(r"(?i)(assistant|<\|assistant\|>|</s>)$", "", sql).strip()

            pattern = r"(?i)\b(FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_\.\"]*)(\s|$)"

            def replace_match(match: re.Match) -> str:
                keyword = match.group(1)
                table_identifier = match.group(2)
                trailing = match.group(3)

                # Strip quotes and schema qualifier for comparison
                table_core = table_identifier.strip('`"').split('.')[-1]

                if table_core != expected_table:
                    return f"{keyword} {expected_table}{trailing}"
                return match.group(0)

            return re.sub(pattern, replace_match, sql)
        except Exception:
            return sql

    def _quote_columns_with_spaces_in_sql(self, sql: str, table_name: str) -> str:
        """Automatically quote column names that contain spaces or non-word chars
        if they appear unquoted in the generated SQL. This reduces errors for
        columns like "primary sales".
        """
        try:
            table_info = self.db_manager.table_schemas.get(table_name, {})
            columns: List[str] = table_info.get("columns", [])
            cols_needing_quotes = [
                col for col in columns
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col)
            ]

            if not cols_needing_quotes:
                return sql

            def replace_outside_quotes(text: str, target: str, replacement: str) -> str:
                result_chars: List[str] = []
                i = 0
                in_single = False
                in_double = False
                length = len(text)
                tlen = len(target)
                while i < length:
                    ch = text[i]
                    if ch == "'" and not in_double:
                        in_single = not in_single
                        result_chars.append(ch)
                        i += 1
                        continue
                    if ch == '"' and not in_single:
                        in_double = not in_double
                        result_chars.append(ch)
                        i += 1
                        continue
                    if not in_single and not in_double and text.startswith(target, i):
                        result_chars.append(replacement)
                        i += tlen
                        continue
                    result_chars.append(ch)
                    i += 1
                return "".join(result_chars)

            updated_sql = sql
            for col in cols_needing_quotes:
                updated_sql = replace_outside_quotes(updated_sql, col, f'"{col}"')
            return updated_sql
        except Exception:
            return sql

    def _enforce_semantic_column_preference(self, nlq: str, sql: str, table_name: str) -> str:
        """If the natural language mentions a specific metric that has a column with
        spaces (e.g., "primary sales"), prefer that column over similarly named ones
        like "sales". Only adjusts when unambiguous and columns exist.
        """
        try:
            table_info = self.db_manager.table_schemas.get(table_name, {})
            columns: List[str] = table_info.get("columns", [])
            nlq_lower = nlq.lower()

            def replace_word_outside_quotes(text: str, word: str, replacement: str) -> str:
                result_chars: List[str] = []
                i = 0
                in_single = False
                in_double = False
                length = len(text)
                wlen = len(word)
                while i < length:
                    ch = text[i]
                    if ch == "'" and not in_double:
                        in_single = not in_single
                        result_chars.append(ch)
                        i += 1
                        continue
                    if ch == '"' and not in_single:
                        in_double = not in_double
                        result_chars.append(ch)
                        i += 1
                        continue
                    if not in_single and not in_double:
                        # Check whole-word match boundaries
                        if i + wlen <= length and text[i:i+wlen].lower() == word.lower():
                            prev_ch = text[i-1] if i > 0 else "\0"
                            next_ch = text[i+wlen] if i + wlen < length else "\0"
                            if not (prev_ch.isalnum() or prev_ch == "_") and not (next_ch.isalnum() or next_ch == "_"):
                                result_chars.append(replacement)
                                i += wlen
                                continue
                    result_chars.append(ch)
                    i += 1
                return "".join(result_chars)

            # Prefer "primary sales" if the user explicitly mentions it and both columns exist
            if ("primary sales" in nlq_lower) and ("primary sales" in columns) and ("sales" in columns):
                sql = replace_word_outside_quotes(sql, "sales", '"primary sales"')

            return sql
        except Exception:
            return sql
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cache_hit_rate = (self.cache.cache_stats["hits"] / 
                         max(1, self.cache.cache_stats["hits"] + self.cache.cache_stats["misses"]))
        
        avg_execution_time = (self.metrics.total_execution_time / 
                             max(1, self.metrics.query_count))
        
        memory_info = config.get_memory_info()
        
        return {
            "query_metrics": {
                "total_queries": self.metrics.query_count,
                "total_execution_time": self.metrics.total_execution_time,
                "average_execution_time": avg_execution_time,
                "errors": self.metrics.errors
            },
            "cache_metrics": {
                "hit_rate": cache_hit_rate,
                "total_hits": self.cache.cache_stats["hits"],
                "total_misses": self.cache.cache_stats["misses"]
            },
            "memory_metrics": {
                "peak_usage_mb": self.metrics.memory_peak_mb,
                "current_usage_mb": memory_monitor.get_memory_usage(),
                "system_memory_gb": memory_info["total_gb"],
                "available_memory_gb": memory_info["available_gb"]
            },
            "loaded_tables": {name: {"rows": info["total_rows"]} 
                            for name, info in self.loaded_tables.items()}
        }


# Global instances
memory_monitor = MemoryMonitor()
nlq_system = None


def _format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with numeric columns formatted as strings
    with thousands separators and fixed decimals for floats.
    This is used only for pretty printing to the console.
    """
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


def initialize_system() -> NLQSystem:
    """Initialize the NLQ system with configuration validation."""
    global nlq_system
    
    if not config.validate():
        raise RuntimeError("Configuration validation failed")
    
    logger.info("Initializing NLQ System...")
    logger.info(f"Configuration: {config.get_memory_info()}")
    
    nlq_system = NLQSystem()
    return nlq_system


def run_query(question: str, table_name: str = "sales_data") -> None:
    """Run a natural language query and display results."""
    if not nlq_system:
        raise RuntimeError("System not initialized. Call initialize_system() first.")
    
    try:
        print(f"\n🔍 Question: {question}")
        print("=" * 60)
        
        result = nlq_system.query(question, table_name)
        
        print(f"\n📜 Generated SQL:")
        print(result.sql_query)
        
        print(f"\n📊 Query Result ({result.row_count:,} rows):")
        if result.row_count > 0:
            # Display results with smart formatting
            if result.row_count <= 20:
                display_df = _format_dataframe_for_display(result.data)
                print(display_df.to_string(index=False))
            else:
                print("First 10 rows:")
                display_df = _format_dataframe_for_display(result.data.head(10))
                print(display_df.to_string(index=False))
                print(f"\n... and {result.row_count - 10:,} more rows")
        else:
            print("No results found.")
        
        print(f"\n⚡ Performance:")
        print(f"  Execution time: {result.execution_time:.3f}s")
        print(f"  Memory usage: {result.memory_usage_mb:.1f}MB")
        print(f"  From cache: {'Yes' if result.from_cache else 'No'}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Query failed: {e}", exc_info=True)


def main():
    """Main function demonstrating the system."""
    try:
        # Initialize system
        system = initialize_system()
        
        # Load data
        print("Loading sales data...")
        if os.path.exists("./data/llm_dataset_v10.gz"):
            load_result = system.load_data("./data/llm_dataset_v10.gz", "sales_data")
            print(f"✅ Loaded {load_result['total_rows']:,} rows in {load_result['duration']:.2f}s")
        else:
            print("❌ sales not found. Please ensure the file exists.")
            return
        
        # Example queries tailored to your schema
        example_queries = [
            # Aggregations and time filters
            "What were the total sales in November 2024 in Central-A?",
            "Show the top 10 cities by total sales in 2024",
            # "What is the total primary sales in Q3 2024?",

            # Dimensional breakdowns
            "List the top 5 brands by sales in Lahore",
            "What are the top 5 customers by sales in 2024?",

            # Booleans and KPIs
            "How many rows had stockouts in 2024?",
            "What is the average mro for productive rows in 2024?",
            "What percent of rows were assortment = TRUE in Central-A?",

            # Mixed filters
            "Total sales for brand CHOCO LAVA in Lahore-A Territory in November 2024",
            "Show the top 5 routes by sales for distributor D0715 in 2024"
        ]
        
        print("\n🚀 Running example queries...")
        for query in example_queries:
            run_query(query)
            time.sleep(1)  # Brief pause between queries
        
        # Performance report
        print("\n📈 Performance Report:")
        report = system.get_performance_report()
        print(json.dumps(report, indent=2))
        
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        print(f"❌ System error: {e}")


if __name__ == "__main__":
    main()
