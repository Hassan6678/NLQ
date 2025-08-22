import os
import time
import logging
from typing import Optional, Dict, Any

import pandas as pd
from llama_cpp import Llama

from config import config


logger = logging.getLogger(__name__)


class LLMManager:
	"""LLM manager for SQL generation and result summarization."""

	def __init__(self):
		self.llm = None
		self.summarizer_llm = None
		self.model_loaded = False
		self._load_model()

	def _load_model(self):
		"""Load main SQL model and the summarizer model (Mistral) if present."""
		try:
			logger.info(f"Loading model: {config.model.model_path}")
			start_time = time.time()
			self.llm = Llama(
				model_path=config.model.model_path,
				n_ctx=config.model.n_ctx,
				n_threads=config.model.n_threads,
				n_gpu_layers=config.model.n_gpu_layers,
				verbose=config.model.verbose,
			)
			logger.info(f"Model loaded successfully in {time.time() - start_time:.2f}s")
			self.model_loaded = True

			# Summarizer model (prefer Mistral path if available)
			try:
				if os.path.exists(config.model.summarizer_model_path) and (
					config.model.summarizer_model_path != config.model.model_path
				):
					logger.info(f"Loading summarizer model: {config.model.summarizer_model_path}")
					s_start = time.time()
					self.summarizer_llm = Llama(
						model_path=config.model.summarizer_model_path,
						n_ctx=max(config.model.n_ctx, 2048),
						n_threads=config.model.n_threads,
						n_gpu_layers=config.model.n_gpu_layers,
						verbose=False,
					)
					logger.info(f"Summarizer model loaded in {time.time() - s_start:.2f}s")
				else:
					self.summarizer_llm = self.llm
			except Exception as se:
				logger.warning(
					f"Failed to load summarizer model, falling back to main model: {se}"
				)
				self.summarizer_llm = self.llm
		except Exception as e:
			logger.error(f"Failed to load model(s): {e}")
			raise

	def build_enhanced_prompt(self, nlq: str, table_info: Dict[str, Any]) -> str:
		"""Build the SQL-generation prompt using table context and examples."""
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
 - Months and years are integers. If the question mentions a quarter (Q1..Q4), map to months: Q1={{1,2,3}}, Q2={{4,5,6}}, Q3={{7,8,9}}, Q4={{10,11,12}}.
 - Boolean flags (e.g., productivity, stockout, assortment) are stored as 0/1 integers. Use predicates like column = 1 or column = 0. Avoid TRUE/FALSE.
 - Do NOT use window functions (e.g., LAG/LEAD) in the WHERE clause. If you need to filter on a window, use QUALIFY or compute in a subquery/CTE and filter in the outer query.
 - Never use placeholder years like 20XX/XXXX or partial years (e.g., 20). Use actual 4-digit years present in the table. If the year isn't specified, prefer the latest available year for the requested month(s).
 - Do NOT add region/city/area/territory/distributor/route filters unless explicitly mentioned in the user's question.

### Dataset information
- The ONLY table to use is "{table_name_for_prompt}" (use this exact name).
- Data is monthly with integer columns: month (1..12), year (e.g., 2024).
- Columns include: region, city, area, territory, distributor, route, customer, sku, brand, variant, packtype, sales, "primary sales", target, mto, productivity, mro, stockout, assortment, and specialized mro components.
- If the question omits months/years, prefer the latest available period(s).

### Supported query types
- Sales/revenue totals and rankings by any dimension.
- Growth analysis (latest vs previous month; or user-specified periods/quarters).
- Targets and gaps: use mto for "missed target"/"target gap" (do not compute target - sales or target - mro). Use SUM() when ranking totals of sales/mro/mto across groups (avoid MAX for these totals).
- Productivity/assortment/stockout counts and percentages (COUNT(DISTINCT customer)/total).
- Time filters: specific months, years, or quarters (Q1..Q4).

### Example patterns (adjust filters and dimensions):
1) Worst productivity percentage by distributor in a city for a month:
SELECT region, city, area, territory, distributor,
  COUNT(DISTINCT CASE WHEN month = 9 AND year = 2024 AND productivity = 1 THEN customer END) AS productive_shops,
  COUNT(DISTINCT CASE WHEN month = 9 AND year = 2024 AND productivity = 0 THEN customer END) AS unproductive_shops,
  COUNT(DISTINCT customer) AS total_shops,
  (COUNT(DISTINCT CASE WHEN month = 9 AND year = 2024 AND productivity = 1 THEN customer END) * 100.0 / COUNT(DISTINCT customer)) AS productivity_percentage,
  (COUNT(DISTINCT CASE WHEN month = 9 AND year = 2024 AND productivity = 0 THEN customer END) * 100.0 / COUNT(DISTINCT customer)) AS unproductivity_percentage
FROM {table_name_for_prompt}
WHERE city = 'lahore'
GROUP BY region, city, area, territory, distributor
ORDER BY unproductivity_percentage DESC
LIMIT 1;

2) Routes with highest stockout percentage in a region for a month:
SELECT region, city, area, territory, distributor, route,
  COUNT(DISTINCT CASE WHEN month = 12 AND year = 2024 AND stockout = 1 THEN customer END) AS stockout_shops,
  COUNT(DISTINCT CASE WHEN month = 12 AND year = 2024 AND stockout = 0 THEN customer END) AS not_stockout_shops,
  COUNT(DISTINCT customer) AS total_shops,
  (COUNT(DISTINCT CASE WHEN month = 12 AND year = 2024 AND stockout = 1 THEN customer END) * 100.0 / COUNT(DISTINCT customer)) AS stockout_percentage
FROM {table_name_for_prompt}
WHERE region = 'north b'
GROUP BY region, city, area, territory, distributor, route
ORDER BY stockout_percentage DESC
LIMIT 1;

3) City growth (replace months with latest/previous if user didn't specify):
SELECT region, city,
  SUM(CASE WHEN month = <latest_month> THEN sales ELSE 0 END) AS sales_current_month,
  SUM(CASE WHEN month = <previous_month> THEN sales ELSE 0 END) AS sales_previous_month,
  ((SUM(CASE WHEN month = <latest_month> THEN sales ELSE 0 END) -
    SUM(CASE WHEN month = <previous_month> THEN sales ELSE 0 END)) /
   NULLIF(SUM(CASE WHEN month = <previous_month> THEN sales ELSE 0 END), 0)) * 100 AS sales_growth_percentage
FROM {table_name_for_prompt}
WHERE city = 'lahore'
GROUP BY region, city
ORDER BY sales_growth_percentage DESC
LIMIT 1;

4) Brands with highest stockout percentage in a region for a month:
SELECT brand,
  COUNT(DISTINCT CASE WHEN month = 11 AND year = 2024 AND stockout = 1 THEN customer END) AS stockout_shops,
  COUNT(DISTINCT customer) AS total_shops,
  (COUNT(DISTINCT CASE WHEN month = 11 AND year = 2024 AND stockout = 1 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS stockout_percentage
FROM {table_name_for_prompt}
WHERE region = 'North-A'
GROUP BY brand
ORDER BY stockout_percentage DESC
LIMIT 5;

5) Route with highest missed target (use mto) within filters:
SELECT route, SUM(mto) AS total_mto
FROM {table_name_for_prompt}
WHERE region = 'South-A' AND month = 12 AND year = 2024
GROUP BY route
ORDER BY total_mto DESC
LIMIT 1;

6) Lowest productive area in a region for a month (percentage):
SELECT area,
  COUNT(DISTINCT CASE WHEN month = 2 AND year = 2024 AND productivity = 1 THEN customer END) AS productive_shops,
  COUNT(DISTINCT customer) AS total_shops,
  (COUNT(DISTINCT CASE WHEN month = 2 AND year = 2024 AND productivity = 1 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS productivity_percentage
FROM {table_name_for_prompt}
WHERE region = 'Central-A'
GROUP BY area
ORDER BY productivity_percentage ASC
LIMIT 1;

### Write a SQL query to answer the question:
# {nlq}

### SQL:
SELECT"""

		return prompt

	def generate_sql(self, prompt: str) -> Optional[str]:
		"""Generate SQL text with extraction and validation."""
		if not self.model_loaded:
			raise RuntimeError("Model not loaded")
		try:
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
					"</s>",
				],
			)
			text = output["choices"][0]["text"]
			return self._extract_and_validate_sql(text)
		except Exception as e:
			logger.error(f"Error generating SQL: {e}")
			return None

	def _extract_and_validate_sql(self, text: str) -> Optional[str]:
		try:
			text = text.strip()
			if "SELECT" not in text.upper():
				if text:
					text = "SELECT " + text
				else:
					return None
			text_upper = text.upper()
			if "SELECT" in text_upper:
				sql_start = text_upper.find("SELECT")
				sql_text = text[sql_start:]
				# End delimiter
				lower_sql_text = sql_text.lower()
				sql_end = len(sql_text)
				for delimiter in [
					";",
					"\n\n",
					"###",
					"assistant",
					"user:",
					"```",
					"<|assistant|>",
					"</s>",
				]:
					pos = lower_sql_text.find(delimiter)
					if pos != -1:
						sql_end = min(sql_end, pos)
				sql = sql_text[:sql_end].strip()
				if not sql.endswith(";"):
					sql += ";"
				if self._validate_sql_syntax(sql):
					return sql
			return None
		except Exception as e:
			logger.error(f"Error extracting SQL: {e}")
			return None

	def _validate_sql_syntax(self, sql: str) -> bool:
		sql_upper = sql.upper()
		if not sql_upper.startswith("SELECT"):
			return False
		for keyword in [" DROP ", " DELETE ", " TRUNCATE ", " ALTER ", " CREATE "]:
			if keyword in f" {sql_upper} ":
				logger.warning(f"Potentially dangerous SQL keyword detected: {keyword}")
				return False
		if sql.count("(") != sql.count(")"):
			return False
		return True

	def summarize_result(self, nlq: str, df: pd.DataFrame, sql: str, max_rows: int = 30) -> str:
		"""Summarize the SQL result using the summarizer model (Mistral preferred)."""
		try:
			if not self.model_loaded:
				return ""
			display_df = df.head(max_rows)
			try:
				table_text = display_df.to_markdown(index=False)
			except Exception:
				table_text = display_df.to_string(index=False)
			prompt = (
				"### You are a senior data analyst.\n"
				"Summarize the result of a SQL query clearly and concisely for a business user.\n\n"
				f"User question:\n{nlq}\n\n"
				f"Result table (first {min(len(df), max_rows)} rows):\n{table_text}\n\n"
				"Instructions:\n"
				"- Provide a short, precise answer in 2-4 sentences.\n"
				"- Highlight the top entity and key metric(s).\n"
				"- If percentages exist, include them.\n"
				"- Do not show numbers in scientific notation. Where large numbers appear, prefer human-readable units (e.g., 1.2 thousand, 3.4 million, 2 billion).\n"
				"- Do not include currency units or symbols (e.g., $, dollar, USD).\n"
				"- If empty, say the filters return no results.\n\n"
				"### Answer:\n"
			)
			engine = self.summarizer_llm if self.summarizer_llm else self.llm
			output = engine(
				prompt,
				temperature=min(1.0, max(0.0, getattr(config.model, "summarizer_temperature", 0.2))),
				max_tokens=min(1024, max(64, getattr(config.model, "summarizer_max_tokens", 384))),
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
					"</s>",
				],
			)
			answer = output["choices"][0]["text"].strip()
			return self._sanitize_summary_text(answer)
		except Exception as e:
			logger.warning(f"summarize_result failed: {e}")
			return ""

	def _sanitize_summary_text(self, text: str) -> str:
		"""Remove currency units/symbols from summaries as per requirements."""
		try:
			import re
			clean = re.sub(r"\$", "", text)
			clean = re.sub(r"\b(dollars?|usd)\b", "", clean, flags=re.IGNORECASE)

			# Convert scientific notation to numeric and format into human units
			def format_to_units(val: float) -> str:
				abs_val = abs(val)
				if abs_val >= 1e12:
					v = val / 1e12
					unit = " trillion"
				elif abs_val >= 1e9:
					v = val / 1e9
					unit = " billion"
				elif abs_val >= 1e6:
					v = val / 1e6
					unit = " million"
				elif abs_val >= 1e3:
					v = val / 1e3
					unit = " thousand"
				else:
					# Avoid scientific notation for small numbers
					if float(int(val)) == val:
						return str(int(val))
					return f"{val:.2f}".rstrip('0').rstrip('.')

				# Format decimal places adaptively
				abs_v = abs(v)
				if abs_v >= 100:
					fmt = f"{v:.0f}"
				elif abs_v >= 10:
					fmt = f"{v:.1f}".rstrip('0').rstrip('.')
				else:
					fmt = f"{v:.2f}".rstrip('0').rstrip('.')
				return fmt + unit

			# Replace scientific notation like 1.23e+06
			def repl_sci(m: re.Match) -> str:
				try:
					num = float(m.group(0))
					return format_to_units(num)
				except Exception:
					return m.group(0)

			clean = re.sub(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)", repl_sci, clean)

			# Replace plain large numbers (with or without commas) except likely years (1900-2100)
			def repl_plain(m: re.Match) -> str:
				s = m.group(0)
				try:
					n = float(s.replace(',', ''))
					# Skip 4-digit years in reasonable range
					if 1900 <= int(n) <= 2100 and len(s.replace(',', '').split('.')[0]) == 4:
						return s
					if abs(n) < 1000:
						# keep as-is but avoid scientific notation
						if float(int(n)) == n:
							return str(int(n))
						return f"{n:.2f}".rstrip('0').rstrip('.')
					return format_to_units(n)
				except Exception:
					return s

			# match numbers with commas or long digits (4+ digits) optionally with decimals
			clean = re.sub(r"(?<![\d,\.\-])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,\.])", repl_plain, clean)
			clean = re.sub(r"(?<![\d\.\-])\d{4,}(?:\.\d+)?(?![\d\.])", repl_plain, clean)

			clean = re.sub(r"\s{2,}", " ", clean).strip()
			return clean
		except Exception:
			return text


