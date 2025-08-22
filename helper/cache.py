import os
import time
import hashlib
import pickle
import logging
from typing import Optional

from config import config
from helper.types import QueryResult


logger = logging.getLogger(__name__)


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
				cache_age = time.time() - os.path.getmtime(cache_file)
				if cache_age < config.cache.cache_ttl:
					with open(cache_file, 'rb') as f:
						result = pickle.load(f)
						result.from_cache = True
						self.cache_stats["hits"] += 1
						logger.debug(f"Result cache hit for SQL: {sql[:50]}...")
						return result
				else:
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


