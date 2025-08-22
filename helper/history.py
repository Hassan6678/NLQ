import os
import json
import time
import threading
from typing import Any, Dict

from config import config


class HistoryManager:
	"""Lightweight thread-safe history logger writing JSONL entries to disk."""

	def __init__(self, history_dir: str = None):
		self.history_dir = history_dir or os.path.join(config.cache.cache_dir, "history")
		os.makedirs(self.history_dir, exist_ok=True)
		self._lock = threading.Lock()

	def _file_path(self) -> str:
		date_str = time.strftime("%Y-%m-%d")
		return os.path.join(self.history_dir, f"history_{date_str}.jsonl")

	def record(self, **entry: Dict[str, Any]) -> None:
		data = {
			"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
			**entry,
		}
		line = json.dumps(data, ensure_ascii=False)
		with self._lock:
			with open(self._file_path(), "a", encoding="utf-8") as f:
				f.write(line + "\n")


history_manager = HistoryManager()


