import os
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple

import duckdb
import pandas as pd

from config import config
from helper.memory import memory_monitor
from helper.types import QueryResult


logger = logging.getLogger(__name__)


class DatabaseManager:
	"""Optimized database manager with connection pooling and chunked processing."""

	def __init__(self):
		self.active_connections: Dict[int, duckdb.DuckDBPyConnection] = {}
		self.active_lock = threading.Lock()
		self.table_schemas: Dict[str, Dict[str, Any]] = {}
		self.local = threading.local()
		self._initialize_database()

	def _init_connection(self) -> duckdb.DuckDBPyConnection:
		"""Initialize a DuckDB connection with proper settings."""
		conn = duckdb.connect(config.database.db_path)
		conn.execute(f"SET memory_limit='{config.database.memory_limit}'")
		conn.execute(f"SET threads={config.database.threads}")
		conn.execute(f"SET enable_progress_bar=false")
		if config.database.enable_parallel:
			conn.execute("PRAGMA threads=4")
		return conn

	def _initialize_database(self):
		"""Ensure database file is valid and log once at startup."""
		try:
			conn = self._init_connection()
			conn.close()

		except Exception as e:
			logger.error(f"Failed to initialize database: {e}", exc_info=True)
			raise

	def get_connection(self):
		"""Return a thread-local connection (creates one if missing)."""
		from contextlib import contextmanager

		@contextmanager
		def _gen():
			if not hasattr(self.local, "conn") or self.local.conn is None:
				self.local.conn = self._init_connection()
			else:
				# Validate connection is still alive
				try:
					self.local.conn.execute("SELECT 1").fetchone()
				except Exception as e:
					logger.warning(f"Connection became invalid, recreating: {e}")
					try:
						self.local.conn.close()
					except:
						pass
					self.local.conn = self._init_connection()
			try:
				yield self.local.conn
			except Exception as e:
				# If connection fails, mark it for recreation
				logger.warning(f"Connection error during use: {e}")
				try:
					if hasattr(self.local, "conn"):
						self.local.conn.close()
				except:
					pass
				self.local.conn = None
				raise
		return _gen()

	def close_all(self):
		"""Close thread-local connection if it exists."""
		if hasattr(self.local, "conn") and self.local.conn is not None:
			try:
				self.local.conn.close()
			except Exception:
				pass
			self.local.conn = None

	def invalidate_all_connections(self):
		"""Force recreation of all connections (after database deletion/recreation)."""
		if hasattr(self.local, "conn") and self.local.conn is not None:
			try:
				self.local.conn.close()
			except Exception:
				pass
			self.local.conn = None

	def clear_schema_cache(self):
		"""Clear all cached table schemas."""
		self.table_schemas.clear()

	def _quote_identifier(self, name: str) -> str:
		if __import__('re').match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
			return name
		return '"' + name.replace('"', '""') + '"'


	def table_exists(self, table_name: str) -> bool:
		"""Check if a table exists in the current DuckDB database file."""
		try:
			with self.get_connection() as conn:
				res = conn.execute("SELECT * FROM information_schema.tables WHERE table_name = ?", [table_name]).fetchone()
			return res is not None
		except Exception:
			return False

	def get_table_row_count(self, table_name: str) -> int:
		try:
			with self.get_connection() as conn:
				res = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
				return int(res[0]) if res else 0
		except Exception:
			return 0

	def load_existing_table_schema(self, table_name: str) -> bool:
		"""Populate table_schemas entry for an already existing table (persisted DB)."""
		try:
			with self.get_connection() as conn:
				desc = conn.execute(f"DESCRIBE {table_name}").fetchall()
				columns = [row[0] for row in desc]
				schema_info = []
				for row in desc:
					col = row[0]
					logical_type = row[1]
					schema_info.append(f"{self._quote_identifier(col)} {logical_type}")
				sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf().to_dict('records')
				self.table_schemas[table_name] = {
					"columns": columns,
					"schema": ", ".join(schema_info),
					"sample_data": sample,
					"table_name": table_name,
				}
			return True
		except Exception as e:
			logger.warning(f"load_existing_table_schema failed: {e}")
			return False

	def load_csv_chunked(self, file_path: str, table_name: str) -> Dict[str, Any]:
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"CSV file not found: {file_path}")

		start_time = time.time()
		total_rows = 0
		with self.get_connection() as conn:
			conn.execute(f"DROP TABLE IF EXISTS {table_name}")
			first_chunk = True
			chunk_count = 0
			for chunk in pd.read_csv(file_path, chunksize=config.database.chunk_size, low_memory=False):
				chunk = self._coerce_chunk_dtypes(chunk)
				chunk_count += 1
				chunk_rows = len(chunk)
				total_rows += chunk_rows
				if first_chunk:
					temp_initial = f"temp_initial_chunk_{int(time.time())}"
					conn.register(temp_initial, chunk)
					conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {temp_initial}")
					conn.execute(f"DROP VIEW IF EXISTS {temp_initial}")
					# Store schema info
					schema_info = []
					for col, dtype in chunk.dtypes.items():
						sql_type = self._pandas_to_sql_type(dtype)
						schema_info.append(f"{self._quote_identifier(col)} {sql_type}")
					self.table_schemas[table_name] = {
						"columns": list(chunk.columns),
						"schema": ", ".join(schema_info),
						"sample_data": chunk.head(3).to_dict('records'),
						"table_name": table_name,
					}
					first_chunk = False
				else:
					temp_table = f"temp_chunk_{chunk_count}"
					conn.register(temp_table, chunk)
					conn.execute(f"INSERT INTO {table_name} SELECT * FROM {temp_table}")
					conn.execute(f"DROP VIEW IF EXISTS {temp_table}")
				if chunk_count % 10 == 0:
					elapsed = time.time() - start_time
					rate = total_rows / elapsed if elapsed > 0 else 0
					if memory_monitor.get_memory_usage() / 1024 > config.get_memory_info()["max_allowed_gb"]:
						logger.warning("Memory usage approaching limit, consider reducing chunk size")
			self._create_indexes(conn, table_name)
		duration = time.time() - start_time

		return {
			"table_name": table_name,
			"total_rows": total_rows,
			"duration": duration,
			"chunks_processed": chunk_count,
			"schema": self.table_schemas[table_name],
		}

	def _pandas_to_sql_type(self, dtype) -> str:
		if pd.api.types.is_integer_dtype(dtype):
			return "INTEGER"
		elif pd.api.types.is_float_dtype(dtype):
			return "DOUBLE"
		elif pd.api.types.is_datetime64_any_dtype(dtype):
			return "TIMESTAMP"
		elif pd.api.types.is_bool_dtype(dtype):
			return "BOOLEAN"
		else:
			return "VARCHAR"

	def _create_indexes(self, conn, table_name: str):
		"""Create indexes on commonly queried columns with better error handling."""
		index_success_count = 0
		index_error_count = 0

		try:
			result = conn.execute(f"DESCRIBE {table_name}").fetchall()
			columns = [row[0] for row in result]

			for col in columns:
				col_lower = col.lower()
				if any(p in col_lower for p in ['id','date','time','region','category','type','month','year','city','brand','sku','customer','territory','route','area','distributor']):
					try:
						conn.execute(f"CREATE INDEX idx_{table_name}_{col} ON {table_name}({col})")
						index_success_count += 1
					except Exception as e:
						logger.warning(f"Could not create index on {col}: {e}")
						index_error_count += 1

			# If many indexes failed, it might indicate a data quality issue
			if index_error_count > index_success_count and index_error_count > 0:
				logger.error(f"High number of index creation failures ({index_error_count}) - check data quality")

		except Exception as e:
			logger.error(f"Error during index creation process: {e}")
			# Don't fail the entire operation for index creation issues

	# Time helpers
	def get_latest_periods(self, table_name: str, limit: int = 2) -> List[Tuple[int, int]]:
		try:
			with self.get_connection() as conn:
				rows = conn.execute(
					f"""
					SELECT year, month FROM {table_name}
					WHERE year IS NOT NULL AND month IS NOT NULL
					GROUP BY year, month
					ORDER BY year DESC, month DESC
					LIMIT {int(limit)}
					"""
				).fetchall()
				return [(int(y), int(m)) for (y, m) in rows]
		except Exception as e:
			logger.warning(f"get_latest_periods failed: {e}")
			return []

	def get_latest_year_for_month(self, table_name: str, month: int) -> Optional[int]:
		try:
			with self.get_connection() as conn:
				row = conn.execute(
					f"SELECT MAX(year) FROM {table_name} WHERE month = ?",
					[int(month)],
				).fetchone()
				return int(row[0]) if row and row[0] is not None else None
		except Exception as e:
			logger.warning(f"get_latest_year_for_month failed: {e}")
			return None

	def get_table_info(self, table_name: str) -> Dict[str, Any]:
		"""Return cached schema with row_count and simple column stats. Best-effort."""
		if table_name not in self.table_schemas:
			return {}
		# If already enriched, return cached
		cached = self.table_schemas.get(table_name, {})
		if cached.get("row_count") is not None and cached.get("column_stats") is not None:
			return cached
		with self.get_connection() as conn:
			try:
				row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
				columns = self.table_schemas[table_name].get("columns", [])
				stats: Dict[str, Any] = {}
				for col in columns[:5]:  # limit to first few columns for performance
					try:
						res = conn.execute(
							f"SELECT COUNT(DISTINCT {col}) AS unique_count, COUNT({col}) AS non_null_count FROM {table_name}"
						).fetchone()
						stats[col] = {"unique_count": res[0], "non_null_count": res[1]}
					except Exception:
						pass
				enriched = {
					**self.table_schemas[table_name],
					"row_count": row_count,
					"column_stats": stats,
				}
				self.table_schemas[table_name] = enriched
				return enriched
			except Exception as e:
				logger.error(f"Error getting table info: {e}")
				return self.table_schemas[table_name]

	def has_columns(self, table_name: str, columns: List[str]) -> bool:
		try:
			with self.get_connection() as conn:
				result = conn.execute(f"DESCRIBE {table_name}").fetchall()
				existing = {row[0].lower() for row in result}
				return all(col.lower() in existing for col in columns)
		except Exception as e:
			logger.warning(f"has_columns failed: {e}")
			return False

	def _coerce_chunk_dtypes(self, chunk: pd.DataFrame) -> pd.DataFrame:
		"""Coerce data types with improved error handling and logging."""
		try:
			coerced = chunk.copy()
			conversion_summary = {"successful": 0, "failed": 0, "warnings": []}

			boolean_candidates = {"productivity", "stockout", "assortment"}

			for col in coerced.columns:
				try:
					series = coerced[col]
					original_dtype = series.dtype

					if col.lower() in {"month", "year"}:
						try:
							coerced[col] = pd.to_numeric(series, errors="coerce").astype("Int64")
							conversion_summary["successful"] += 1

						except Exception as e:
							conversion_summary["warnings"].append(f"Failed to convert {col} to Int64: {e}")
							conversion_summary["failed"] += 1
						continue

					if col.lower() in boolean_candidates:
						try:
							str_vals = series.astype(str).str.strip().str.lower()
							true_set = {"true","1","yes","y","t"}
							false_set = {"false","0","no","n","f"}
							mapped = str_vals.map(lambda v: 1 if v in true_set else (0 if v in false_set else pd.NA))
							if mapped.notna().mean() > 0.9:
								coerced[col] = mapped.astype("Int64")
								conversion_summary["successful"] += 1
								continue
							numeric = pd.to_numeric(series, errors="coerce")
							if numeric.notna().mean() > 0.9 and set(numeric.dropna().unique()).issubset({0,1}):
								coerced[col] = numeric.astype("Int64")
								conversion_summary["successful"] += 1
								continue
							if pd.api.types.is_bool_dtype(series):
								coerced[col] = series.astype("Int64")
								conversion_summary["successful"] += 1
								continue
						except Exception as e:
							conversion_summary["warnings"].append(f"Failed boolean conversion for {col}: {e}")
							conversion_summary["failed"] += 1

					if pd.api.types.is_bool_dtype(series):
						continue
					if pd.api.types.is_numeric_dtype(series):
						continue

					if pd.api.types.is_object_dtype(series):
						try:
							cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
							numeric = pd.to_numeric(cleaned, errors="coerce")
							non_null_ratio = numeric.notna().mean()
							if non_null_ratio > 0.9:
								if (numeric.dropna() % 1 == 0).all():
									coerced[col] = numeric.astype("Int64")
									conversion_summary["successful"] += 1
								else:
									coerced[col] = numeric.astype("float64")
									conversion_summary["successful"] += 1
						except Exception as e:
							conversion_summary["warnings"].append(f"Failed numeric conversion for {col}: {e}")
							conversion_summary["failed"] += 1

				except Exception as e:
					conversion_summary["warnings"].append(f"Unexpected error processing column {col}: {e}")
					conversion_summary["failed"] += 1			

			return coerced

		except Exception as e:
			logger.error(f"Critical error in data type coercion: {e}")
			logger.warning("Falling back to original chunk without type coercion")
			return chunk

	def execute_query(self, sql: str) -> QueryResult:
		start_time = time.time()
		thread_id = threading.get_ident()
		with self.get_connection() as conn:
			try:
				with self.active_lock:
					self.active_connections[thread_id] = conn
				result_df = conn.execute(sql).fetchdf()
				execution_time = time.time() - start_time
				memory_usage = memory_monitor.get_memory_usage()
				return QueryResult(
					data=result_df,
					sql_query=sql,
					execution_time=execution_time,
					row_count=len(result_df),
					memory_usage_mb=memory_usage,
				)
			except Exception as e:
				logger.error(f"SQL execution error: {e}")
				logger.error(f"SQL: {sql}")
				raise
			finally:
				with self.active_lock:
					self.active_connections.pop(thread_id, None)

	def cancel_query_for_thread(self, thread_id: Optional[int] = None) -> bool:
		try:
			tid = thread_id if thread_id is not None else threading.get_ident()
			with self.active_lock:
				conn = self.active_connections.get(tid)
			if conn is None:
				return False
			if hasattr(conn, "interrupt"):
				try:
					conn.interrupt()
					return True
				except Exception as e:
					logger.warning(f"interrupt() failed: {e}")
			try:
				conn.close()
				return True
			except Exception as e:
				logger.warning(f"close() failed during cancel: {e}")
				return False
		except Exception as e:
			logger.warning(f"cancel_query_for_thread failed: {e}")
			return False


