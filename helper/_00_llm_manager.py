import os
import time
import logging
from typing import Optional, Dict, Any

import pandas as pd
from llama_cpp import Llama
# import torch

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
		# gpu_available = torch.cuda.is_available()
		# n_gpu_layers = config.model.n_gpu_layers if gpu_available else 0
		# print(f"{'Using GPU' if gpu_available else 'Using CPU'} for model loading")
		try:
			logger.info(f"Loading model: {config.model.model_path}")
			start_time = time.time()   
			   
			 # Main SQL model
			self.llm = Llama(
				model_path=config.model.model_path,
				n_ctx=config.model.n_ctx,
				n_threads=config.model.n_threads,
				n_gpu_layers=config.model.n_gpu_layers,
				# n_gpu_layers=n_gpu_layers,
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
						# n_gpu_layers=n_gpu_layers,
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

### CRITICAL DUCKDB CONSTRAINTS:
- Use ONLY DuckDB-compatible SQL syntax
- NEVER use window functions like LAG(), LEAD(), ROW_NUMBER(), RANK(), DENSE_RANK()
- NEVER use OVER() clauses with PARTITION BY or ORDER BY
- For growth/trend analysis, use CASE statements with month comparisons instead
- Use CTEs (WITH clauses) for complex calculations when needed

### IMPORTANT RULE TO AVOID HALLUCINATIONS FROM EXAMPLES/SAMPLES:
- Do NOT use values appearing in the sample data/context as implicit filters. Never invent filters (region/city/brand/etc.) based on sample rows unless the NLQ explicitly requests them.

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
 - **CRITICAL**: For region queries, distinguish between:
   - **General region queries** (e.g., "top 5 regions", "all regions", "highest sales by region"): Do NOT add any region filters, just GROUP BY region
   - **Specific region queries** (e.g., "sales in Central-A", "performance in North region"): Add the exact region filter using `LOWER(region) = 'region-name'`
 - If the user asks a general question about totals or rankings (e.g., "highest sales", "top performance") without specifying a dimension, provide a high-level summary by `region`. Do not add other filters unless specified.
 - For city-specific questions, use a simple `WHERE LOWER(city) = '...'` filter. Do not use complex joins if the information is in the same table. Only add this if the user clearly mentions a specific city name in the question — the user may name a city without the word "city" (e.g., "lahore"). If the NLQ contains a city token present in the dataset, include the city filter; otherwise do not invent one.
 - For all string comparisons in `WHERE` clauses, use the `LOWER()` function for case-insensitive matching (e.g., `LOWER(city) = 'karachi'`).
 - When the NLQ explicitly mentions a location (region/city/area/territory/distributor/route), exclude rows where `route = 'UNK'` from sales summaries by adding `route <> 'UNK'` to the WHERE clause.
 - If no location is mentioned in the NLQ, do not automatically add `route <> 'UNK'` or any location filters; return totals over the full dataset (subject to time defaults).

### Dataset information
- The ONLY table to use is "{table_name_for_prompt}" (use this exact name).
- Data is monthly with integer columns: month (1..12), year (e.g., 2024).
- Columns include: region, city, area, territory, distributor, route, customer, sku, brand, variant, packtype, sales, "primary sales", target, mto, productivity, mro, stockout, assortment, and specialized mro components.
- If the user specifies explicit months and/or year in the question, USE THOSE EXACT PERIODS in the SQL. Do NOT change or infer different months/years.
- If the question omits months/years, prefer the latest available period(s).
- **Do NOT invent or add any dimension filters (region/city/area/territory/distributor/route/brand/sku/customer) that are not explicitly present in the user's question.** Only include such filters when the NLQ contains clear, explicit mentions.
- Region to city mapping: Central-A is lahore, South-C is hyderabad, Central-B is faisalabad, Central-D is gujranwala, Central-C is multan, North-B is peshawar, North-A is rawalpindi, South-B is sukkur, South-A is karachi.

### Supported query types
- Sales/revenue totals and rankings by any dimension.
- Growth analysis (latest vs previous month; or user-specified periods/quarters).
- Targets and gaps: use mto for "missed target"/"target gap" (do not compute target - sales or target - mro). Use SUM() when ranking totals of sales/mro/mto across groups (avoid MAX for these totals).
- Productivity/assortment/stockout counts and percentages (COUNT(DISTINCT customer)/total).
  When the NLQ asks for "highest stockouts" by some dimension in a period and optional region, compute:
  - stockout_count = SUM(CASE WHEN stockout = 1 THEN 1 ELSE 0 END)
  - stockout_percentage = stockout_count * 100.0 / NULLIF(COUNT(*), 0)
  And order by stockout_percentage DESC, then stockout_count DESC.
 - Stockout analysis: For "highest stockouts" by a dimension (e.g., brands) in a period and region,
   compute both count and percentage in one query, e.g.:
   SELECT brand,
          SUM(CASE WHEN stockout = 1 THEN 1 ELSE 0 END) AS stockout_count,
          COUNT(*) AS total_records,
          (SUM(CASE WHEN stockout = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0)) AS stockout_percentage
   FROM {table_name_for_prompt}
   WHERE LOWER(region) = 'north-a' AND month = 7 AND year = 2025
   GROUP BY brand
   ORDER BY stockout_percentage DESC, stockout_count DESC
   LIMIT 5;
- Time filters: specific months, years, or quarters (Q1..Q4).

### Example patterns (adjust filters and dimensions):
1) Highest sales route in Karachi for a specific year:
SELECT route, SUM(sales) AS total_sales
FROM {table_name_for_prompt}
WHERE LOWER(city) = 'karachi' AND year = 2025 AND route <> 'UNK'
GROUP BY route
ORDER BY total_sales DESC
LIMIT 1;

2) Worst productivity percentage by distributor in a city for a month:
SELECT region, city, area, territory, distributor,
  COUNT(DISTINCT CASE WHEN month = 9 AND year = 2024 AND productivity = 1 THEN customer END) AS productive_shops,
  COUNT(DISTINCT CASE WHEN month = 9 AND year = 2024 AND productivity = 0 THEN customer END) AS unproductive_shops,
  COUNT(DISTINCT customer) AS total_shops,
  (COUNT(DISTINCT CASE WHEN month = 9 AND year = 2024 AND productivity = 1 THEN customer END) * 100.0 / COUNT(DISTINCT customer)) AS productivity_percentage,
  (COUNT(DISTINCT CASE WHEN month = 9 AND year = 2024 AND productivity = 0 THEN customer END) * 100.0 / COUNT(DISTINCT customer)) AS unproductivity_percentage
FROM {table_name_for_prompt}
WHERE LOWER(city) = 'lahore' AND route <> 'UNK'
GROUP BY region, city, area, territory, distributor
ORDER BY unproductivity_percentage DESC
LIMIT 1;

3) Routes with highest stockout percentage in a region for a month:
SELECT region, city, area, territory, distributor, route,
  COUNT(DISTINCT CASE WHEN month = 12 AND year = 2024 AND stockout = 1 THEN customer END) AS stockout_shops,
  COUNT(DISTINCT CASE WHEN month = 12 AND year = 2024 AND stockout = 0 THEN customer END) AS not_stockout_shops,
  COUNT(DISTINCT customer) AS total_shops,
  (COUNT(DISTINCT CASE WHEN month = 12 AND year = 2024 AND stockout = 1 THEN customer END) * 100.0 / COUNT(DISTINCT customer)) AS stockout_percentage
FROM {table_name_for_prompt}
WHERE LOWER(region) = 'north b' AND route <> 'UNK'
GROUP BY region, city, area, territory, distributor, route
ORDER BY stockout_percentage DESC
LIMIT 1;

4) City growth (use user-specified periods as-is; otherwise use latest vs previous):
SELECT region, city,
  SUM(CASE WHEN month = <latest_month> THEN sales ELSE 0 END) AS sales_current_month,
  SUM(CASE WHEN month = <previous_month> THEN sales ELSE 0 END) AS sales_previous_month,
  ((SUM(CASE WHEN month = <latest_month> THEN sales ELSE 0 END) -
    SUM(CASE WHEN month = <previous_month> THEN sales ELSE 0 END)) /
   NULLIF(SUM(CASE WHEN month = <previous_month> THEN sales ELSE 0 END), 0)) * 100 AS sales_growth_percentage
FROM {table_name_for_prompt}
WHERE LOWER(city) = 'lahore' AND route <> 'UNK'
GROUP BY region, city
ORDER BY sales_growth_percentage DESC
LIMIT 1;

5) Brands with highest stockout percentage in a region for a month:
SELECT brand,
  COUNT(DISTINCT CASE WHEN month = 11 AND year = 2024 AND stockout = 1 THEN customer END) AS stockout_shops,
  COUNT(DISTINCT customer) AS total_shops,
  (COUNT(DISTINCT CASE WHEN month = 11 AND year = 2024 AND stockout = 1 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS stockout_percentage
FROM {table_name_for_prompt}
WHERE LOWER(region) = 'north-a' AND route <> 'UNK'
GROUP BY brand
ORDER BY stockout_percentage DESC
LIMIT 5;

6) Route with highest missed target (use mto) within filters:
SELECT route, SUM(mto) AS total_mto
FROM {table_name_for_prompt}
WHERE LOWER(region) = 'south-a' AND month = 12 AND year = 2024 AND route <> 'UNK'
GROUP BY route
ORDER BY total_mto DESC
LIMIT 1;

7) Lowest productive area in Central-A region for a month (specific region):
SELECT area,
  COUNT(DISTINCT CASE WHEN month = 2 AND year = 2024 AND productivity = 1 THEN customer END) AS productive_shops,
  COUNT(DISTINCT customer) AS total_shops,
  (COUNT(DISTINCT CASE WHEN month = 2 AND year = 2024 AND productivity = 1 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS productivity_percentage
FROM {table_name_for_prompt}
WHERE LOWER(region) = 'central-a' AND route <> 'UNK'
GROUP BY area
ORDER BY productivity_percentage ASC
LIMIT 1;

8) Region with most growth potential (comparing two months):
WITH monthly_sales AS (
    SELECT region, month, SUM(sales) AS total_sales
    FROM {table_name_for_prompt}
    WHERE year = 2024 AND route <> 'UNK'
    GROUP BY region, month
)
SELECT region,
       (SUM(CASE WHEN month = 12 THEN total_sales END) -
        SUM(CASE WHEN month = 11 THEN total_sales END)) * 100.0 /
        NULLIF(SUM(CASE WHEN month = 11 THEN total_sales END), 0) AS sales_growth_percentage
FROM monthly_sales
GROUP BY region
ORDER BY sales_growth_percentage DESC
LIMIT 1;

9) Top 5 regions by sales in a specific month (general region query - NO region filters):
SELECT region, SUM(sales) AS total_sales
FROM {table_name_for_prompt}
WHERE month = 1 AND year = 2025 AND route <> 'UNK'
GROUP BY region
ORDER BY total_sales DESC
LIMIT 5;

10) Territory growth analysis (avoiding window functions):
SELECT territory,
       SUM(CASE WHEN month = 12 AND year = 2024 THEN sales ELSE 0 END) AS sales_december,
       SUM(CASE WHEN month = 11 AND year = 2024 THEN sales ELSE 0 END) AS sales_november,
       ((SUM(CASE WHEN month = 12 AND year = 2024 THEN sales ELSE 0 END) -
         SUM(CASE WHEN month = 11 AND year = 2024 THEN sales ELSE 0 END)) /
        NULLIF(SUM(CASE WHEN month = 11 AND year = 2024 THEN sales ELSE 0 END), 0)) * 100 AS growth_percentage
FROM {table_name_for_prompt}
WHERE route <> 'UNK'
GROUP BY territory
HAVING SUM(CASE WHEN month = 11 AND year = 2024 THEN sales ELSE 0 END) > 0
ORDER BY growth_percentage DESC
LIMIT 5;

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


