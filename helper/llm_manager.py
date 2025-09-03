import os
import time
import logging
import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import pandas as pd
from llama_cpp import Llama

from config import config

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    model_path: str
    summarizer_model_path: Optional[str]
    n_ctx: int
    n_threads: int
    n_gpu_layers: int
    temperature: float
    max_tokens: int
    summarizer_temperature: float
    summarizer_max_tokens: int
    verbose: bool


@dataclass
class SQLValidationResult:
    """Result of SQL validation."""
    is_valid: bool
    error_message: Optional[str] = None
    sanitized_sql: Optional[str] = None


class LLMManager:
    """
    Enhanced LLM manager for SQL generation and result summarization.

    Features:
    - Robust model loading with fallback mechanisms
    - Intelligent SQL generation with validation
    - Optimized result summarization
    - Comprehensive error handling and logging
    - Performance monitoring and caching
    """

    def __init__(self):
        self.llm: Optional[Llama] = None
        self.summarizer_llm: Optional[Llama] = None
        self.model_loaded = False
        self._model_config = self._build_model_config()
        self._load_model()



    def _build_model_config(self) -> ModelConfig:
        """Build model configuration from config with validation."""
        return ModelConfig(
            model_path=config.model.model_path,
            summarizer_model_path=getattr(config.model, 'summarizer_model_path', None),
            n_ctx=getattr(config.model, 'n_ctx', 4096),
            n_threads=getattr(config.model, 'n_threads', 8),
            n_gpu_layers=getattr(config.model, 'n_gpu_layers', 0),
            temperature=getattr(config.model, 'temperature', 0.1),
            max_tokens=getattr(config.model, 'max_tokens', 512),
            summarizer_temperature=getattr(config.model, 'summarizer_temperature', 0.2),
            summarizer_max_tokens=getattr(config.model, 'summarizer_max_tokens', 384),
            verbose=getattr(config.model, 'verbose', False)
        )

    def _load_model(self) -> None:
        """Load main SQL model and summarizer model with enhanced error handling."""
        try:
            if not os.path.exists(self._model_config.model_path):
                raise FileNotFoundError(f"Model file not found: {self._model_config.model_path}")

            # Load main SQL model
            self.llm = Llama(
                model_path=self._model_config.model_path,
                n_ctx=self._model_config.n_ctx,
                n_threads=self._model_config.n_threads,
                n_gpu_layers=self._model_config.n_gpu_layers,
                verbose=self._model_config.verbose,
            )

            # Load summarizer model if available
            self._load_summarizer_model()

            self.model_loaded = True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _load_summarizer_model(self) -> None:
        """Load summarizer model with fallback to main model."""
        if not self._model_config.summarizer_model_path:
            self.summarizer_llm = self.llm
            return

        try:
            if os.path.exists(self._model_config.summarizer_model_path) and (
                self._model_config.summarizer_model_path != self._model_config.model_path
            ):
                self.summarizer_llm = Llama(
                    model_path=self._model_config.summarizer_model_path,
                    n_ctx=max(self._model_config.n_ctx, 2048),
                    n_threads=self._model_config.n_threads,
                    n_gpu_layers=self._model_config.n_gpu_layers,
                    verbose=False,
                )
            else:
                self.summarizer_llm = self.llm

        except Exception as e:
            logger.warning(f"Failed to load summarizer model, falling back to main model: {e}")
            self.summarizer_llm = self.llm

    def build_enhanced_prompt(self, nlq: str, table_info: Dict[str, Any]) -> str:
        """
        Build enhanced SQL generation prompt with comprehensive context and validation.

        Args:
            nlq: Natural language query
            table_info: Dictionary containing table schema, sample data, and statistics

        Returns:
            Formatted prompt string for LLM
        """
        self._validate_table_info(table_info)

        schema = table_info.get("schema", "")
        sample_data = table_info.get("sample_data", [])
        row_count = table_info.get("row_count", 0)
        table_name = table_info.get("table_name", "sales_data")

        # Build context sections
        sample_context = self._build_sample_context(sample_data)
        stats_context = self._build_statistics_context(table_info, row_count)

        # Calculate available tokens for SQL generation
        prompt_base_length = len(f"""### You are an expert SQL generator for DuckDB.
        ### Task: Generate a single, correct DuckDB SQL query that answers the user's question.
        ### Given the following table schema and context:

        # Table: {table_name}
        # Schema: {schema}
        # Total rows: {row_count:,}
        {sample_context}
        {stats_context}""")

        # Reserve at least 1024 tokens for SQL generation
        available_tokens = max(512, self._model_config.n_ctx - (prompt_base_length // 4) - 256)

        prompt = f"""### You are an expert SQL generator for DuckDB.
        ### Task: Generate a single, COMPLETE DuckDB SQL query that answers the user's question.
        ### IMPORTANT: Your response MUST start with 'SELECT' and be a valid, executable SQL statement.
        ### Given the following table schema and context:

        # Table: {table_name}
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
        - The ONLY available table is named "{table_name}". You must use exactly this table name in all FROM and JOIN clauses. Do not invent any other table names.
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
        - The ONLY table to use is "{table_name}" (use this exact name).
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
        - **CRITICAL: For percentage queries involving customer-level metrics (productivity, assortment, stockout), ALWAYS use COUNT(DISTINCT customer) to count unique shops/customers:**
        - Productivity percentage: Use COUNT(DISTINCT customer) for both productive and unproductive shops
        - Assortment percentage: Use COUNT(DISTINCT customer) for both assorted and unassorted shops
        - Stockout percentage: Use COUNT(DISTINCT customer) for both stockout and non-stockout shops
        - **NEVER use COUNT(*) for these percentage calculations** - always use COUNT(DISTINCT customer)
        - For other metrics (sales, mro, mto, etc.), use COUNT(*) or SUM() as appropriate.
        - Stockout analysis: For "highest stockouts" by a dimension (e.g., brands) in a period and region,
        compute both count and percentage using COUNT(DISTINCT customer) for unique shops, e.g.:
        SELECT brand,
                COUNT(DISTINCT CASE WHEN stockout = 1 THEN customer END) AS stockout_shops,
                COUNT(DISTINCT CASE WHEN stockout = 0 THEN customer END) AS not_stockout_shops,
                COUNT(DISTINCT customer) AS total_shops,
                (COUNT(DISTINCT CASE WHEN stockout = 1 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS stockout_percentage,
                (COUNT(DISTINCT CASE WHEN stockout = 0 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS not_stockout_percentage
        FROM {table_name}
        WHERE LOWER(region) = 'north-a' AND month = 7 AND year = 2025
        GROUP BY brand
        ORDER BY stockout_percentage DESC, stockout_shops DESC
        LIMIT 5;
        - Time filters: specific months, years, or quarters (Q1..Q4).

        ### Example patterns (adjust filters and dimensions):
        1) Highest sales route in Karachi for a specific year:
        SELECT route, SUM(sales) AS total_sales
        FROM {table_name}
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
        FROM {table_name}
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
        FROM {table_name}
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
        FROM {table_name}
        WHERE LOWER(city) = 'lahore' AND route <> 'UNK'
        GROUP BY region, city
        ORDER BY sales_growth_percentage DESC
        LIMIT 1;

        5) Brands with highest stockout percentage in a region for a month:
        SELECT brand,
        COUNT(DISTINCT CASE WHEN month = 11 AND year = 2024 AND stockout = 1 THEN customer END) AS stockout_shops,
        COUNT(DISTINCT customer) AS total_shops,
        (COUNT(DISTINCT CASE WHEN month = 11 AND year = 2024 AND stockout = 1 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS stockout_percentage
        FROM {table_name}
        WHERE LOWER(region) = 'north-a' AND route <> 'UNK'
        GROUP BY brand
        ORDER BY stockout_percentage DESC
        LIMIT 5;

        6) Route with highest missed target (use mto) within filters:
        SELECT route, SUM(mto) AS total_mto
        FROM {table_name}
        WHERE LOWER(region) = 'south-a' AND month = 12 AND year = 2024 AND route <> 'UNK'
        GROUP BY route
        ORDER BY total_mto DESC
        LIMIT 1;

        7) Lowest productive area in Central-A region for a month (specific region):
        SELECT area,
        COUNT(DISTINCT CASE WHEN month = 2 AND year = 2024 AND productivity = 1 THEN customer END) AS productive_shops,
        COUNT(DISTINCT customer) AS total_shops,
        (COUNT(DISTINCT CASE WHEN month = 2 AND year = 2024 AND productivity = 1 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS productivity_percentage
        FROM {table_name}
        WHERE LOWER(region) = 'central-a' AND route <> 'UNK'
        GROUP BY area
        ORDER BY productivity_percentage ASC
        LIMIT 1;

        8) Region with most growth potential (comparing two months):
        WITH monthly_sales AS (
        SELECT region, month, SUM(sales) AS total_sales
        FROM {table_name}
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
        FROM {table_name}
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
        FROM {table_name}
        WHERE route <> 'UNK'
        GROUP BY territory
        HAVING SUM(CASE WHEN month = 11 AND year = 2024 THEN sales ELSE 0 END) > 0
        ORDER BY growth_percentage DESC
        LIMIT 5;

        ### Write a SQL query to answer the question:
        {nlq}

        ### IMPORTANT: Generate ONLY the complete SQL query. Do NOT include explanations.
        ### SQL (DuckDB):
        SELECT"""

        return prompt

    def _validate_table_info(self, table_info: Dict[str, Any]) -> None:
        """Validate table information before prompt building."""
        if not isinstance(table_info, dict):
            raise ValueError("table_info must be a dictionary")

        required_keys = ['schema', 'table_name']
        for key in required_keys:
            if key not in table_info:
                logger.warning(f"Missing recommended key in table_info: {key}")

    def _build_sample_context(self, sample_data: List[Dict]) -> str:
        """Build sample data context section."""
        if not sample_data:
            return ""

        context = "\n### Sample data (first 3 rows):\n"
        for i, row in enumerate(sample_data[:3], 1):
            context += f"Row {i}: {row}\n"
        return context

    def _build_statistics_context(self, table_info: Dict[str, Any], row_count: int) -> str:
        """Build column statistics context section."""
        column_stats = table_info.get("column_stats", {})
        if not column_stats or row_count == 0:
            return ""

        context = "\n### Column information:\n"
        for col, stats in column_stats.items():
            unique_pct = (stats["unique_count"] / row_count * 100) if row_count > 0 else 0
            context += ".1f"
        return context

    def generate_sql(self, prompt: str) -> Optional[str]:
        """
        Generate SQL with enhanced extraction and validation.

        Args:
            prompt: Formatted prompt for LLM

        Returns:
            Validated SQL string or None if generation fails
        """
        if not self.model_loaded:
            logger.error("❌ LLM generation failed: Model not loaded")
            raise RuntimeError("Model not loaded - cannot generate SQL")

        if not prompt or not isinstance(prompt, str):
            logger.error("❌ LLM generation failed: Invalid prompt provided")
            raise ValueError("Invalid prompt provided")

        try:
            # Calculate dynamic max tokens based on prompt length and context window
            prompt_tokens = len(prompt) // 4 if prompt else 0
            remaining_tokens = self._model_config.n_ctx - prompt_tokens - 100  # Reserve 100 tokens for safety
            dynamic_max_tokens = min(self._model_config.max_tokens, max(512, remaining_tokens))

            if remaining_tokens < 256:  # Not enough space for meaningful SQL
                return None

            output = self.llm(
                prompt,
                temperature=self._model_config.temperature,
                max_tokens=dynamic_max_tokens,
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
                    ";",  # Allow semicolon as a natural stopping point
                ],
            )

            raw_text = output["choices"][0]["text"]

            # Check if we got a meaningful response
            if not raw_text or len(raw_text.strip()) == 0:
                return None

            # Check for common failure patterns
            if any(phrase in raw_text.lower() for phrase in ["i don't know", "i cannot", "i'm sorry", "unable to"]):
                logger.warning("LLM indicated uncertainty in response")
                # Don't fail here, let SQL extraction handle it

            # Enhanced validation: Check if response looks like it could be SQL
            raw_text_stripped = raw_text.strip()
            if not raw_text_stripped.upper().startswith("SELECT") and not raw_text_stripped[0].isupper():
                # Try to fix incomplete responses by prepending SELECT
                if raw_text_stripped and not raw_text_stripped.upper().startswith(("SELECT", "FROM", "WHERE", "GROUP", "ORDER")):
                    raw_text = f"SELECT {raw_text}"

            result = self._extract_and_validate_sql(raw_text)
            if not result:
                # Try to fix incomplete SQL with a focused retry
                fixed_result = self._try_fix_incomplete_sql(raw_text)
                if fixed_result:
                    result = fixed_result
                else:
                    logger.error("Failed to extract valid SQL from LLM response")

            return result

        except MemoryError as e:
            logger.error(f"Memory error: {e}")
            return None

        except OSError as e:
            if "context" in str(e).lower() or "token" in str(e).lower():
                logger.error(f"Context window exceeded: {e}")
                return None
            else:
                logger.error(f"OS/Model loading error: {e}")
                return None

        except KeyError as e:
            logger.error(f"Unexpected response format: {e}")
            return None

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None



    def _extract_and_validate_sql(self, text: str) -> Optional[str]:
        """
        Extract and validate SQL from LLM output with enhanced parsing.

        Args:
            text: Raw text from LLM

        Returns:
            Validated SQL string or None
        """
        try:
            if not text or not isinstance(text, str):
                logger.error("❌ SQL extraction failed: Input text is empty or not a string")
                return None

            text = text.strip()
            if not text:
                return None

            # Ensure SELECT is present - more flexible approach
            if "SELECT" not in text.upper():
                text_stripped = text.strip()
                if text_stripped.startswith(("FROM", "WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "JOIN")):
                    text = "SELECT " + text
                elif "," in text_stripped and ("FROM" in text_stripped.upper() or "WHERE" in text_stripped.upper()):
                    # Looks like column list without SELECT
                    text = "SELECT " + text
                elif text_stripped and not text_stripped[0].isupper() and len(text_stripped.split()) > 2:
                    # Looks like partial SQL starting with lowercase
                    text = "SELECT " + text
                else:
                    logger.error("SQL extraction failed: No SELECT keyword found in response")
                    return None

            text_upper = text.upper()

            # Find SQL start
            if "SELECT" in text_upper:
                sql_start = text_upper.find("SELECT")
                sql_text = text[sql_start:]
            else:
                logger.error("SQL extraction failed: Could not locate SELECT statement")
                return None

            # Find SQL end using multiple delimiters
            lower_sql_text = sql_text.lower()
            sql_end = len(sql_text)

            delimiters = [
                ";",
                "\n\n",
                "###",
                "assistant",
                "user:",
                "```",
                "<|assistant|>",
                "</s>",
            ]

            for delimiter in delimiters:
                pos = lower_sql_text.find(delimiter)
                if pos != -1:
                    sql_end = min(sql_end, pos)
                    break

            sql = sql_text[:sql_end].strip()

            # Ensure semicolon
            if not sql.endswith(";"):
                sql += ";"

            # Validate SQL
            validation = self._validate_sql_syntax(sql)
            if validation.is_valid:
                final_sql = validation.sanitized_sql or sql
                return final_sql

            logger.error(f"SQL validation failed: {validation.error_message}")
            return None

        except Exception as e:
            logger.error(f"❌ SQL extraction failed: Unexpected error during parsing - {e}")
            logger.error(f"💡 Raw LLM response: {text[:500]}...")
            return None

    def _validate_sql_syntax(self, sql: str) -> SQLValidationResult:
        """
        Enhanced SQL validation with detailed error reporting.

        Args:
            sql: SQL string to validate

        Returns:
            SQLValidationResult with validation status and details
        """
        try:
            if not sql or not isinstance(sql, str):
                return SQLValidationResult(False, "Empty or invalid SQL")

            sql_upper = sql.upper()

            # Must start with SELECT
            if not sql_upper.startswith("SELECT"):
                return SQLValidationResult(False, "SQL must start with SELECT")

            # Check for dangerous keywords
            dangerous_keywords = [" DROP ", " DELETE ", " TRUNCATE ", " ALTER ", " CREATE ", " INSERT ", " UPDATE "]
            for keyword in dangerous_keywords:
                if keyword in f" {sql_upper} ":
                    return SQLValidationResult(False, f"Potentially dangerous SQL keyword detected: {keyword.strip()}")

            # Check balanced parentheses
            if sql.count("(") != sql.count(")"):
                return SQLValidationResult(False, "Unbalanced parentheses")

            # Basic structure validation
            if ";" not in sql:
                return SQLValidationResult(False, "SQL must end with semicolon")

            return SQLValidationResult(True)

        except Exception as e:
            return SQLValidationResult(False, f"Validation error: {str(e)}")

    def summarize_result(self, nlq: str, df: pd.DataFrame, sql: str, max_rows: int = 30) -> str:
        """
        Enhanced result summarization with better error handling and formatting.

        Args:
            nlq: Original natural language query
            df: Result DataFrame
            sql: Generated SQL query
            max_rows: Maximum rows to include in summary

        Returns:
            Human-readable summary string
        """
        try:
            if not self.model_loaded:
                logger.warning("Model not loaded, cannot summarize")
                return ""

            if df is None or df.empty:
                return "No results found for the query."

            if not isinstance(df, pd.DataFrame):
                return "Invalid data format for summarization."

            # Validate inputs
            if not nlq or not isinstance(nlq, str):
                nlq = "Query analysis"

            if not sql or not isinstance(sql, str):
                sql = "Generated SQL"

            # Prepare data for summarization
            display_df = df.head(max_rows)

            try:
                table_text = display_df.to_markdown(index=False)
            except Exception:
                table_text = display_df.to_string(index=False)

            # Build summarization prompt
            prompt = self._build_summarization_prompt(nlq, table_text, len(df), max_rows)

            # Generate summary
            summary = self._generate_summary(prompt)

            # Sanitize and format
            return self._sanitize_summary_text(summary)

        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return ""

    def _build_summarization_prompt(self, nlq: str, table_text: str, total_rows: int, max_rows: int) -> str:
        """Build enhanced summarization prompt."""
        return (
            "### You are a senior data analyst.\n"
            "Summarize the result of a SQL query clearly and concisely for a business user.\n\n"
            f"User question:\n{nlq}\n\n"
            f"Result table (first {min(total_rows, max_rows)} of {total_rows} rows):\n{table_text}\n\n"
            "Instructions:\n"
            "- Provide a short, precise answer in 2-4 sentences.\n"
            "- Highlight the top entity and key metric(s).\n"
            "- If percentages exist, include them.\n"
            "- Do not show numbers in scientific notation. Where large numbers appear, prefer human-readable units (e.g., 1.2 thousand, 3.4 million, 2 billion).\n"
            "- Do not include currency units or symbols (e.g., $, dollar, USD).\n"
            "- If empty, say the filters return no results.\n\n"
            "### Answer:\n"
        )

    def _generate_summary(self, prompt: str) -> str:
        """Generate summary using the appropriate model."""
        engine = self.summarizer_llm if self.summarizer_llm else self.llm

        output = engine(
            prompt,
            temperature=self._model_config.summarizer_temperature,
            max_tokens=self._model_config.summarizer_max_tokens,
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

        return output["choices"][0]["text"].strip()

    def _sanitize_summary_text(self, text: str) -> str:
        """
        Enhanced text sanitization for summaries with better number formatting.

        Args:
            text: Raw summary text

        Returns:
            Sanitized and formatted text
        """
        try:
            if not text:
                return ""

            import re

            # Remove currency symbols and units
            text = re.sub(r"\$", "", text)
            text = re.sub(r"\b(dollars?|usd)\b", "", text, flags=re.IGNORECASE)

            # Format large numbers
            text = self._format_large_numbers(text)

            # Clean up whitespace
            text = re.sub(r"\s{2,}", " ", text).strip()

            return text

        except Exception as e:
            logger.warning(f"Text sanitization failed: {e}")
            return text

    def _format_large_numbers(self, text: str) -> str:
        """Format large numbers into human-readable units."""
        def format_to_units(match):
            try:
                num = float(match.group(0).replace(',', ''))
                abs_val = abs(num)

                if abs_val >= 1e12:
                    val = num / 1e12
                    unit = " trillion"
                elif abs_val >= 1e9:
                    val = num / 1e9
                    unit = " billion"
                elif abs_val >= 1e6:
                    val = num / 1e6
                    unit = " million"
                elif abs_val >= 1e3:
                    val = num / 1e3
                    unit = " thousand"
                else:
                    # Keep smaller numbers as-is, avoiding unnecessary scientific notation
                    if float(int(num)) == num:
                        return str(int(num))
                    return ".2f"

                # Adaptive decimal formatting
                abs_v = abs(val)
                if abs_v >= 100:
                    fmt = ".0f"
                elif abs_v >= 10:
                    fmt = ".1f"
                else:
                    fmt = ".2f"

                # Remove unnecessary zeros
                formatted = f"{val:{fmt}}".rstrip('0').rstrip('.')
                return formatted + unit

            except (ValueError, ZeroDivisionError):
                return match.group(0)

        def repl_sci(match):
            """Replace scientific notation."""
            try:
                num = float(match.group(0))
                return format_to_units(match)
            except Exception:
                return match.group(0)

        # Replace scientific notation
        text = re.sub(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)", repl_sci, text)

        # Replace plain large numbers (with or without commas)
        # Skip 4-digit years in reasonable range
        def repl_plain(match):
            s = match.group(0)
            try:
                num = float(s.replace(',', ''))
                # Skip 4-digit years
                if 1900 <= int(num) <= 2100 and len(s.replace(',', '').split('.')[0]) == 4:
                    return s
                if abs(num) < 1000:
                    # Keep small numbers clean
                    if float(int(num)) == num:
                        return str(int(num))
                    return ".2f"
                return format_to_units(match)
            except Exception:
                return s

        # Match numbers with optional commas and decimals
        text = re.sub(r"(?<![\d,\.\-])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,\.])", repl_plain, text)
        text = re.sub(r"(?<![\d\.\-])\d{4,}(?:\.\d+)?(?![\d\.])", repl_plain, text)

        return text

    def _try_fix_incomplete_sql(self, raw_text: str) -> Optional[str]:
        """Attempt to fix incomplete or malformed SQL responses."""
        try:
            text = raw_text.strip()

            # Case 1: Response starts with column names (like "city, SUM(sales)")
            if "," in text[:50] and any(keyword in text.upper() for keyword in ["FROM", "WHERE", "GROUP", "ORDER"]):
                # Try to extract components
                parts = text.split()
                from_idx = next((i for i, part in enumerate(parts) if part.upper() == "FROM"), -1)
                if from_idx > 0:
                    columns = " ".join(parts[:from_idx])
                    rest = " ".join(parts[from_idx:])
                    fixed_sql = f"SELECT {columns} {rest}"
                    if not fixed_sql.endswith(";"):
                        fixed_sql += ";"
                    return fixed_sql

            # Case 2: Missing SELECT but has FROM clause
            if text.upper().startswith(("FROM", "WHERE")):
                fixed_sql = f"SELECT * {text}"
                if not fixed_sql.endswith(";"):
                    fixed_sql += ";"
                return fixed_sql

            # Case 3: Response looks like partial column specification
            if text and not text[0].isupper() and len(text.split()) >= 3:
                # Check if it looks like a column aggregation pattern
                if any(agg in text.upper() for agg in ["SUM(", "COUNT(", "AVG(", "MAX(", "MIN("]):
                    fixed_sql = f"SELECT {text}"
                    if not fixed_sql.endswith(";"):
                        fixed_sql += ";"
                    return fixed_sql

            # Case 4: Try to append missing components
            if "FROM" in text.upper() and not text.upper().startswith("SELECT"):
                # Look for table name after FROM
                from_match = text.upper().find("FROM")
                table_part = text[from_match:]
                fixed_sql = f"SELECT * {table_part}"
                if not fixed_sql.endswith(";"):
                    fixed_sql += ";"
                return fixed_sql

            return None

        except Exception as e:
            logger.warning(f"🔧 SQL repair failed: {e}")
            return None

    def is_ready(self) -> bool:
        """Check if the LLM manager is ready for use."""
        return self.model_loaded and self.llm is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "main_model_loaded": self.llm is not None,
            "summarizer_model_loaded": self.summarizer_llm is not None,
            "uses_separate_summarizer": self.summarizer_llm is not self.llm,
            "model_path": self._model_config.model_path,
            "summarizer_path": self._model_config.summarizer_model_path,
            "context_window": self._model_config.n_ctx,
            "threads": self._model_config.n_threads
        }