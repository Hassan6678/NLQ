import psutil
from contextlib import contextmanager
import time
from typing import Generator


class MemoryMonitor:
	def __init__(self):
		self.peak_usage = 0.0
		self.current_usage = 0.0

	def get_memory_usage(self) -> float:
		process = psutil.Process()
		usage_mb = process.memory_info().rss / 1024 / 1024
		self.current_usage = usage_mb
		self.peak_usage = max(self.peak_usage, usage_mb)
		return usage_mb

	@contextmanager
	def monitor_operation(self, operation_name: str) -> Generator[None, None, None]:
		from config import config
		start_memory = self.get_memory_usage()
		start_time = time.time()
		try:
			yield
		finally:
			end_memory = self.get_memory_usage()
			duration = time.time() - start_time
			memory_delta = end_memory - start_memory
			if getattr(config.logging, "enable_performance_logging", True):
				import logging
				logger = logging.getLogger(__name__)
				logger.info(f"{operation_name} - Duration: {duration:.2f}s, Memory delta: {memory_delta:.1f}MB, Peak: {self.peak_usage:.1f}MB")


memory_monitor = MemoryMonitor()


