from dataclasses import dataclass
import pandas as pd


@dataclass
class QueryResult:
	data: pd.DataFrame
	sql_query: str
	execution_time: float
	row_count: int
	from_cache: bool = False
	memory_usage_mb: float = 0.0
	summary_text: str = ""


@dataclass
class PerformanceMetrics:
	query_count: int = 0
	total_execution_time: float = 0.0
	errors: int = 0
	cache_hits: int = 0
	cache_misses: int = 0
	memory_peak_mb: float = 0.0
	total_execution_time: float = 0.0
	cache_hits: int = 0
	cache_misses: int = 0
	memory_peak_mb: float = 0.0
	errors: int = 0


