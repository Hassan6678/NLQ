import time
import os
import re
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from config import config
from helper.db import DatabaseManager
from helper.llm_manager import LLMManager
from helper.cache import QueryCache
from helper.types import QueryResult, PerformanceMetrics
from helper.memory import memory_monitor

logger = logging.getLogger(__name__)


class QueryMode(Enum):
    """Query processing modes."""
    SUM_METRIC = "sum_metric"
    STOCKOUT_PERCENT = "stockout_percent"
    GROWTH_PERCENT = "growth_percent"
    MTO_SUM = "mto_sum"
    PRODUCTIVITY_PERCENT = "productivity_percent"
    ASSORTMENT_PERCENT = "assortment_percent"


@dataclass
class Intent:
    """Structured representation of user query intent."""
    dimension: str
    metric: str
    metric_mode: QueryMode
    year: Optional[int] = None
    months: List[int] = field(default_factory=list)
    order_dir: str = "DESC"
    limit: int = 5
    filters: List[Tuple[str, str, Any]] = field(default_factory=list)
    is_growth: bool = False


class IntentExtractor:
    """Handles extraction of user intent from natural language queries."""

    MONTH_MAP = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
    }

    QUARTER_MAP = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}

    def extract_periods_from_nlq(self, nlq_lower: str) -> Tuple[Optional[int], List[int]]:
        """Extract year and months from natural language query."""
        year: Optional[int] = None
        months: List[int] = []

        # Extract year
        year_match = re.search(r"\b(20\d{2})\b", nlq_lower)
        if year_match:
            try:
                year = int(year_match.group(1))
            except ValueError:
                pass

        # Extract quarters
        quarter_match = re.search(r"\bq([1-4])\b", nlq_lower)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            months.extend(self.QUARTER_MAP[quarter])

        # Extract month names
        for name, num in self.MONTH_MAP.items():
            if re.search(rf"\b{name}\b", nlq_lower):
                months.append(num)

        # Extract numeric months
        for match in re.findall(r"\bmonth\s*(=|is|:)\s*(\d{1,2})\b", nlq_lower):
            try:
                months.append(int(match[1]))
            except ValueError:
                pass

        # Deduplicate and validate months
        months = sorted(set(m for m in months if 1 <= m <= 12))

        return year, months

    def extract_intent_from_nlq(self, nlq: str, table_cols: List[str], table_name: str,
                               db_manager: DatabaseManager) -> Optional[Intent]:
        """Parse NLQ into structured intent."""
        try:
            text = nlq.lower().strip()

            # Extract dimension
            dimension = self._extract_dimension(text, table_cols)
            if not dimension:
                return None

            # Determine query type and analysis mode
            is_growth = self._is_growth_query(text)
            wants_stockout = self._wants_stockout_analysis(text, table_cols)
            wants_productivity = self._wants_productivity_analysis(text, table_cols)
            wants_assortment = self._wants_assortment_analysis(text, table_cols)
            wants_missed_target = self._wants_missed_target_analysis(text, table_cols)

            # Extract metric and mode
            metric, metric_mode = self._extract_metric_and_mode(
                text, table_cols, is_growth, wants_stockout, wants_productivity,
                wants_assortment, wants_missed_target
            )
            if not metric:
                return None

            # Extract time periods
            year, months = self.extract_periods_from_nlq(text)

            # Fill missing year if months specified
            if months and year is None:
                try:
                    year = db_manager.get_latest_year_for_month(table_name, max(months))
                except Exception:
                    periods = db_manager.get_latest_periods(table_name, 1)
                    year = periods[0][0] if periods else None

            # Extract latest periods if no time specified
            if year is None and not months:
                try:
                    periods = db_manager.get_latest_periods(table_name, 1)
                    if periods:
                        year, month = periods[0]
                        months = [month]
                except Exception:
                    pass

            # Extract ordering and limit
            order_dir = "ASC" if self._is_ascending_query(text) else "DESC"
            limit = self._extract_limit(text)

            # Extract filters
            filters = self._extract_filters(text, table_cols, db_manager, table_name)

            return Intent(
                dimension=dimension,
                metric=metric,
                metric_mode=metric_mode,
                year=year,
                months=months,
                order_dir=order_dir,
                limit=limit,
                filters=filters,
                is_growth=is_growth
            )

        except Exception as e:
            logger.warning(f"Intent extraction failed: {e}")
            return None

    def _extract_dimension(self, text: str, table_cols: List[str]) -> Optional[str]:
        """
        Extract grouping dimension from query text with comprehensive pattern matching.
        Handles various user expressions like:
        - "5 cities", "top 3 cities", "lowest 5 cities"
        - "cities with highest stockout", "best performing cities"
        - "which city", "show me cities"
        """
        candidate_dims = ["region", "city", "area", "territory", "distributor", "route", "brand", "sku", "customer"]
        available_dims = [d for d in candidate_dims if d in table_cols]

        # Comprehensive patterns to find dimensions in various contexts
        dimension_patterns = [
            # Pattern 1: Number followed by dimension (e.g., "5 cities", "3 regions")
            r"(\d+)\s+(\w+)",

            # Pattern 2: Qualifier + number + dimension (e.g., "top 3 cities", "bottom 5 regions")
            r"(top|bottom|lowest|highest|best|worst|upper|lower|first|last)\s+\d+\s+(\w+)",

            # Pattern 3: Dimension after qualifiers (e.g., "highest sales cities", "best performing regions")
            r"(highest|lowest|best|worst|top|bottom|upper|lower|first|last|most|least)\s+(\w+)",

            # Pattern 4: Question words + dimension (e.g., "which city", "what region")
            r"(which|what|show\s+me)\s+(\w+)",

            # Pattern 5: Dimension before qualifiers (e.g., "cities with highest", "regions by sales")
            r"(\w+)\s+(with|by|having|for|in)",

            # Pattern 6: Simple dimension mentions (fallback)
            r"(\w+)"
        ]

        for i, pattern in enumerate(dimension_patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Handle different pattern structures
                if i == 0:  # "(\d+)\s+(\w+)" -> (number, word)
                    potential_dim = match[1].lower()
                elif i == 1:  # "(top|bottom|lowest|highest|best|worst|upper|lower|first|last)\s+\d+\s+(\w+)" -> (qualifier, word)
                    potential_dim = match[1].lower()
                elif i in [2, 3, 4]:  # Patterns with (qualifier, word) or (question, word) or (word, preposition)
                    potential_dim = match[1].lower()
                else:  # Pattern 6: single word
                    potential_dim = match.lower()

                # Check if it's a valid dimension
                result_dim = self._validate_and_normalize_dimension(potential_dim, available_dims)
                if result_dim:
                    return result_dim

        # Fallback: look for any exact dimension match
        for dim in available_dims:
            if re.search(rf"\b{dim}\b", text, re.IGNORECASE):
                return dim

        # Default to region if available
        default_dim = "region" if "region" in table_cols else (available_dims[0] if available_dims else None)
        return default_dim

    def _validate_and_normalize_dimension(self, potential_dim: str, available_dims: List[str]) -> Optional[str]:
        """
        Validate if a potential dimension is valid and normalize plural forms.
        Returns the normalized dimension name or None if invalid.
        """
        # Exact match
        if potential_dim in available_dims:
            return potential_dim

        # Handle plural forms
        if potential_dim.endswith('s') and potential_dim[:-1] in available_dims:
            return potential_dim[:-1]

        if potential_dim + 's' in available_dims:
            return potential_dim + 's'

        # Handle common variations
        variations = {
            'route': ['routes'],
            'city': ['cities'],
            'region': ['regions'],
            'area': ['areas'],
            'territory': ['territories'],
            'distributor': ['distributors'],
            'brand': ['brands'],
            'sku': ['skus'],
            'customer': ['customers']
        }

        for standard_dim, variant_list in variations.items():
            if potential_dim in variant_list and standard_dim in available_dims:
                return standard_dim

        return None

    def _is_growth_query(self, text: str) -> bool:
        """Determine if query is asking for growth analysis."""
        growth_keywords = [
            "growth", "increase", "decrease", "change", "trend", "compare",
            "previous", "last month", "month over month", "mom", "yoy",
            "year over year", "compared to", "vs", "versus"
        ]
        return any(keyword in text for keyword in growth_keywords)

    def _wants_stockout_analysis(self, text: str, table_cols: List[str]) -> bool:
        """Check if query wants stockout analysis."""
        return ("stockout" in text or "stock out" in text or "out of stock" in text) and ("stockout" in table_cols)

    def _wants_productivity_analysis(self, text: str, table_cols: List[str]) -> bool:
        """Check if query wants productivity percentage analysis."""
        productivity_terms = ["productivity", "productive", "unproductive", "unproductivity", "productiveness"]
        return any(term in text for term in productivity_terms) and ("productivity" in table_cols)

    def _wants_assortment_analysis(self, text: str, table_cols: List[str]) -> bool:
        """Check if query wants assortment percentage analysis."""
        assortment_terms = ["assortment", "assorted", "unassorted", "assort", "assorting"]
        return any(term in text for term in assortment_terms) and ("assortment" in table_cols)

    def _wants_missed_target_analysis(self, text: str, table_cols: List[str]) -> bool:
        """Check if query wants missed target analysis."""
        return any(term in text for term in ["missed target", "target missed", "target gap", "gap to target", "mto"]) and ("mto" in table_cols)

    def _extract_metric_and_mode(self, text: str, table_cols: List[str], is_growth: bool,
                                wants_stockout: bool, wants_productivity: bool,
                                wants_assortment: bool, wants_missed_target: bool) -> Tuple[Optional[str], QueryMode]:
        """Extract metric and determine query mode."""
        if wants_stockout:
            return "stockout", QueryMode.STOCKOUT_PERCENT
        elif wants_productivity:
            return "productivity", QueryMode.PRODUCTIVITY_PERCENT
        elif wants_assortment:
            return "assortment", QueryMode.ASSORTMENT_PERCENT
        elif is_growth:
            metric = "sales" if "sales" in table_cols else ("mro" if "mro" in table_cols else None)
            return metric, QueryMode.GROWTH_PERCENT
        elif wants_missed_target:
            return "mto", QueryMode.MTO_SUM
        else:
            metric = self._choose_metric_from_nlq(text, table_cols)
            return metric, QueryMode.SUM_METRIC

    def _choose_metric_from_nlq(self, nlq_lower: str, table_cols: List[str]) -> Optional[str]:
        """Choose appropriate metric based on query content."""
        metric_patterns = {
            "mro": ["growth opportunity", "growth potential", "potential to grow", "potential",
                   "potential improvement", "missed revenue", "missed gain", "mro", "opportunity"],
            "mto": ["missed target", "target missed", "sales shortfall", "performance gap", "mto"],
            "primary sales": ["primary sales", "primary contribution"],
            "target": ["target"],
            "sales": ["sales", "revenue", "contribution"]
        }

        for metric, patterns in metric_patterns.items():
            if metric in table_cols and any(pattern in nlq_lower for pattern in patterns):
                return metric

        # Default to sales if available
        return "sales" if "sales" in table_cols else None

    def _is_ascending_query(self, text: str) -> bool:
        """Check if query wants ascending order (worst, lowest, etc.)."""
        asc_keywords = ["least", "lowest", "worst", "minimum", "min", "bottom"]
        return any(keyword in text for keyword in asc_keywords)

    def _extract_limit(self, text: str) -> int:
        """
        Extract limit number from query with comprehensive pattern matching.
        Handles various expressions like:
        - "5 cities", "top 3 regions", "lowest 5 cities"
        - "which city", "highest performing region"
        - "all regions", "show me cities"
        """
        # Check for "all" keyword that implies no limit (or high limit)
        if "all" in text.lower():
            return 20  # Show all (reasonable limit)

        # Pattern 1: Qualifier + number patterns (e.g., "top 3", "bottom 5", "lowest 10")
        qualifier_num_patterns = [
            r"(top|bottom|lowest|highest|best|worst|upper|lower|first|last)\s+(\d+)",
            r"(\d+)\s+(top|bottom|lowest|highest|best|worst|upper|lower|first|last)"
        ]

        for pattern in qualifier_num_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract the number (could be in group 1 or 2 depending on pattern)
                for group in match.groups():
                    if group and group.isdigit():
                        limit = max(1, int(group))
                        return limit

        # Pattern 2: Just a number followed by dimension (e.g., "5 cities", "3 regions")
        number_dim_pattern = r"(\d+)\s+(\w+)"
        match = re.search(number_dim_pattern, text, re.IGNORECASE)
        if match:
            limit = max(1, int(match.group(1)))
            return limit

        # Pattern 3: Single result indicators (which, what, highest without number)
        single_result_keywords = [
            "which", "what", "highest", "best", "maximum", "max",
            "lowest", "worst", "minimum", "min", "top", "bottom"
        ]

        # Check if any single result keyword appears without a number
        for keyword in single_result_keywords:
            if re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
                # Check if there's no number after this keyword
                keyword_pattern = rf"{keyword}\s*(\d*)"
                keyword_match = re.search(keyword_pattern, text, re.IGNORECASE)
                if keyword_match and not keyword_match.group(1):  # No number after keyword
                    return 1

        # Pattern 4: Question words that imply single answers
        question_patterns = [r"which\s+\w+", r"what\s+\w+", r"show\s+me\s+\w+"]
        for pattern in question_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 1

        # Default to showing top 5 for general queries
        return 5

    def _extract_filters(self, text: str, table_cols: List[str],
                        db_manager: DatabaseManager, table_name: str) -> List[Tuple[str, str, Any]]:
        """Extract filters from query text."""
        filters: List[Tuple[str, str, Any]] = []

        # Region filters
        region_filter = self._extract_region_filter(text, table_cols)
        if region_filter:
            filters.append(region_filter)

        return filters

    def _extract_region_filter(self, text: str, table_cols: List[str]) -> Optional[Tuple[str, str, str]]:
        """Extract region filter from query."""
        if "region" not in table_cols:
            return None

        # Exact region match (e.g., Central-A, North B, South-A)
        exact_match = re.search(r"\b(north|south|central|east|west)[\-\s]?([ab])\b", text, re.IGNORECASE)
        if exact_match:
            region_name = f"{exact_match.group(1).capitalize()}-{exact_match.group(2).upper()}"
            return ("region", "=", region_name)

        # Partial region match (e.g., "Central region")
        partial_match = re.search(r"\b(north|south|central|east|west)\s+(region|area|territory)\b", text, re.IGNORECASE)
        if partial_match:
            base_region = partial_match.group(1).capitalize()
            return ("region", "LIKE", f"{base_region}-%")

        return None


class SQLGenerator:
    """Handles generation of SQL queries from intents."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def build_sql_from_intent(self, intent: Intent, table_name: str) -> str:
        """
        Build SQL query from structured intent.

        Args:
            intent: Query intent
            table_name: Target table name

        Returns:
            Generated SQL query
        """
        dim_sql = self.db_manager._quote_identifier(intent.dimension)
        metric_sql = self.db_manager._quote_identifier(intent.metric)

        # Build WHERE clause
        where_parts = self._build_where_parts(intent)
        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""

        # Generate SQL based on mode
        if intent.metric_mode == QueryMode.STOCKOUT_PERCENT:
            sql = self._build_stockout_sql(dim_sql, table_name, where_clause, intent)
        elif intent.metric_mode == QueryMode.PRODUCTIVITY_PERCENT:
            sql = self._build_productivity_sql(dim_sql, table_name, where_clause, intent)
        elif intent.metric_mode == QueryMode.ASSORTMENT_PERCENT:
            sql = self._build_assortment_sql(dim_sql, table_name, where_clause, intent)
        elif intent.metric_mode == QueryMode.GROWTH_PERCENT:
            sql = self._build_growth_sql(intent, table_name)
        elif intent.metric_mode == QueryMode.MTO_SUM:
            sql = self._build_mto_sql(dim_sql, metric_sql, table_name, where_clause, intent)
        else:  # SUM_METRIC
            sql = self._build_sum_sql(dim_sql, metric_sql, table_name, where_clause, intent)

        return sql

    def _build_where_parts(self, intent: Intent) -> List[str]:
        """Build WHERE clause components from intent."""
        where_parts = []

        if intent.year:
            where_parts.append(f"year = {intent.year}")

        if intent.months:
            month_list = ", ".join(str(m) for m in sorted(set(intent.months)))
            where_parts.append(f"month IN ({month_list})")

        for col, op, val in intent.filters:
            col_sql = self.db_manager._quote_identifier(col)
            if isinstance(val, str):
                if op.upper() == "LIKE":
                    where_parts.append(f"{col_sql} LIKE '{val}'")
                else:
                    where_parts.append(f"LOWER({col_sql}) = '{val.lower()}'")
            else:
                where_parts.append(f"{col_sql} {op} {val}")

        return where_parts

    def _build_stockout_sql(self, dim_sql: str, table_name: str, where_clause: str, intent: Intent) -> str:
        """Build stockout percentage analysis SQL using unique customer counts."""
        return (
            f"SELECT {dim_sql}, "
            f"COUNT(DISTINCT CASE WHEN stockout = 1 THEN customer END) AS stockout_shops, "
            f"COUNT(DISTINCT CASE WHEN stockout = 0 THEN customer END) AS not_stockout_shops, "
            f"COUNT(DISTINCT customer) AS total_shops, "
            f"(COUNT(DISTINCT CASE WHEN stockout = 1 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS stockout_percentage, "
            f"(COUNT(DISTINCT CASE WHEN stockout = 0 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS not_stockout_percentage "
            f"FROM {table_name}{where_clause} "
            f"GROUP BY {dim_sql} "
            f"ORDER BY stockout_percentage {intent.order_dir}, stockout_shops {intent.order_dir} "
            f"LIMIT {intent.limit};"
        )

    def _build_productivity_sql(self, dim_sql: str, table_name: str, where_clause: str, intent: Intent) -> str:
        """Build productivity percentage analysis SQL using unique customer counts."""
        return (
            f"SELECT {dim_sql}, "
            f"COUNT(DISTINCT CASE WHEN productivity = 1 THEN customer END) AS productive_shops, "
            f"COUNT(DISTINCT CASE WHEN productivity = 0 THEN customer END) AS un_productive_shops, "
            f"COUNT(DISTINCT customer) AS total_shops, "
            f"(COUNT(DISTINCT CASE WHEN productivity = 1 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS productive_percentage, "
            f"(COUNT(DISTINCT CASE WHEN productivity = 0 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS un_productive_percentage "
            f"FROM {table_name}{where_clause} "
            f"GROUP BY {dim_sql} "
            f"ORDER BY productive_percentage {intent.order_dir}, productive_shops {intent.order_dir} "
            f"LIMIT {intent.limit};"
        )

    def _build_assortment_sql(self, dim_sql: str, table_name: str, where_clause: str, intent: Intent) -> str:
        """Build assortment percentage analysis SQL using unique customer counts."""
        return (
            f"SELECT {dim_sql}, "
            f"COUNT(DISTINCT CASE WHEN assortment = 1 THEN customer END) AS assorted_shops, "
            f"COUNT(DISTINCT CASE WHEN assortment = 0 THEN customer END) AS un_assorted_shops, "
            f"COUNT(DISTINCT customer) AS total_shops, "
            f"(COUNT(DISTINCT CASE WHEN assortment = 1 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS assorted_percentage, "
            f"(COUNT(DISTINCT CASE WHEN assortment = 0 THEN customer END) * 100.0 / NULLIF(COUNT(DISTINCT customer), 0)) AS un_assorted_percentage "
            f"FROM {table_name}{where_clause} "
            f"GROUP BY {dim_sql} "
            f"ORDER BY assorted_percentage {intent.order_dir}, assorted_shops {intent.order_dir} "
            f"LIMIT {intent.limit};"
        )

    def _build_growth_sql(self, intent: Intent, table_name: str) -> str:
        """Build growth analysis SQL."""
        dim_sql = self.db_manager._quote_identifier(intent.dimension)
        metric_sql = self.db_manager._quote_identifier(intent.metric)

        if not intent.months:
            # Use latest two periods
            periods = self.db_manager.get_latest_periods(table_name, limit=2)
            if len(periods) >= 2:
                year, month1 = periods[0]
                _, month2 = periods[1]
            else:
                year, month1 = periods[0]
                month2 = month1 - 1 if month1 > 1 else 12
        else:
            # Use specified months
            year = intent.year
            month1, month2 = sorted(intent.months)[-2:] if len(intent.months) >= 2 else (intent.months[0], intent.months[0] - 1)

        return (
            f"SELECT {dim_sql}, "
            f"SUM(CASE WHEN month = {month1} THEN {metric_sql} ELSE 0 END) AS {intent.metric}_month_{month1}, "
            f"SUM(CASE WHEN month = {month2} THEN {metric_sql} ELSE 0 END) AS {intent.metric}_month_{month2}, "
            f"((SUM(CASE WHEN month = {month1} THEN {metric_sql} ELSE 0 END) - "
            f"SUM(CASE WHEN month = {month2} THEN {metric_sql} ELSE 0 END)) / "
            f"NULLIF(SUM(CASE WHEN month = {month2} THEN {metric_sql} ELSE 0 END), 0)) * 100 AS growth_percentage "
            f"FROM {table_name} "
            f"WHERE year = {year} AND month IN ({month2}, {month1}) "
            f"GROUP BY {dim_sql} "
            f"HAVING SUM(CASE WHEN month = {month2} THEN {metric_sql} ELSE 0 END) > 0 "
            f"ORDER BY growth_percentage {intent.order_dir} "
            f"LIMIT {intent.limit};"
        )

    def _build_mto_sql(self, dim_sql: str, metric_sql: str, table_name: str, where_clause: str, intent: Intent) -> str:
        """Build missed target (MTO) analysis SQL."""
        return (
            f"SELECT {dim_sql}, SUM({metric_sql}) AS total_mto "
            f"FROM {table_name}{where_clause} "
            f"GROUP BY {dim_sql} "
            f"ORDER BY total_mto {intent.order_dir} "
            f"LIMIT {intent.limit};"
        )

    def _build_sum_sql(self, dim_sql: str, metric_sql: str, table_name: str, where_clause: str, intent: Intent) -> str:
        """Build standard sum aggregation SQL."""
        alias_metric = re.sub(r"[^A-Za-z0-9_]+", "_", intent.metric)
        return (
            f"SELECT {dim_sql}, SUM({metric_sql}) AS total_{alias_metric} "
            f"FROM {table_name}{where_clause} "
            f"GROUP BY {dim_sql} "
            f"ORDER BY total_{alias_metric} {intent.order_dir} "
            f"LIMIT {intent.limit};"
        )


class NLQSystem:
    """
    Enhanced NLQ (Natural Language Query) system with modular architecture.

    Features:
    - Intent extraction and SQL generation
    - Robust SQL repair and optimization
    - Intelligent caching and performance monitoring
    - Comprehensive error handling
    """

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.llm_manager = LLMManager()
        self.cache = QueryCache()
        self.metrics = PerformanceMetrics()

        # Initialize modular components
        self.intent_extractor = IntentExtractor()
        self.sql_generator = SQLGenerator(self.db_manager)

        self.loaded_tables = {}

    # ---------------- Data Management -----------------
    def fast_attach_existing(self, table_name: str = "sales_data") -> bool:
        """Attach existing table without reloading."""
        if self.db_manager.table_exists(table_name):
            if self.db_manager.load_existing_table_schema(table_name):
                row_count = self.db_manager.get_table_row_count(table_name)
                self.loaded_tables[table_name] = {
                    "table_name": table_name,
                    "total_rows": row_count,
                    "duration": 0.0,
                    "chunks_processed": 0,
                    "schema": self.db_manager.table_schemas.get(table_name, {})
                }
                return True
        return False

    def delete_database(self) -> bool:
        """Clean database for rebuild."""
        try:
            path = config.database.db_path
            self.db_manager.close_all()
            if os.path.exists(path):
                os.remove(path)
            self.db_manager._initialize_connections()
            self.loaded_tables.clear()
            self.cache.query_cache.clear()
            return True
        except Exception as e:
            logger.error(f"Database deletion failed: {e}")
            return False

    def reset_table(self, table_name: str = "sales_data") -> bool:
        """Drop single table."""
        try:
            if not self.db_manager.table_exists(table_name):
                return True
            with self.db_manager.get_connection() as conn:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.loaded_tables.pop(table_name, None)
            self.db_manager.table_schemas.pop(table_name, None)
            return True
        except Exception as e:
            logger.warning(f"Table reset failed: {e}")
            return False

    def load_data(self, file_path: str, table_name: str = "sales_data") -> Dict[str, Any]:
        """Load data with monitoring."""
        with memory_monitor.monitor_operation(f"Loading {file_path}"):
            result = self.db_manager.load_csv_chunked(file_path, table_name)
            self.loaded_tables[table_name] = result

            # Initialize latest periods in LLM manager for default values
            try:
                latest_periods = self.db_manager.get_latest_periods(table_name, limit=5)
                if latest_periods:
                    # Only update if method exists (optional functionality)
                    if hasattr(self.llm_manager, 'update_latest_periods'):
                        self.llm_manager.update_latest_periods(latest_periods)
            except Exception as e:
                logger.warning(f"Could not initialize latest periods: {e}")

            return result

    # ---------------- Main Query Processing -----------------
    def query(self, nlq: str, table_name: str = "sales_data") -> QueryResult:
        """
        Process natural language query with LLM-first pipeline and robust fallbacks.

        Args:
            nlq: Natural language query
            table_name: Target table name

        Returns:
            QueryResult with data and metadata
        """
        start_time = time.time()

        try:
            # Validation
            if table_name not in self.loaded_tables:
                raise ValueError(f"Table {table_name} not loaded")

            # Check cache first
            cached_sql = self.cache.get_cached_sql(nlq)
            used_llm = False
            sql_source = "cache" if cached_sql else None
            if cached_sql:
                sql = cached_sql
            else:
                # LLM-first generation with rule-based fallback (configurable)
                if getattr(config.system, "prefer_llm_first", True):

                    try:
                        # Update latest periods in LLM manager before generating prompt
                        latest_periods = self.db_manager.get_latest_periods(table_name, limit=3)
                        if latest_periods:
                            # Only update if method exists (optional functionality)
                            if hasattr(self.llm_manager, 'update_latest_periods'):
                                self.llm_manager.update_latest_periods(latest_periods)
                            logger.debug(f"📅 Updated LLM with latest periods: {[f'{m}/{y}' for y, m in latest_periods[:2]]}")

                        table_info = self.db_manager.get_table_info(table_name)
                        prompt = self.llm_manager.build_enhanced_prompt(nlq, table_info)

                        sql = self.llm_manager.generate_sql(prompt)

                        if sql:
                            used_llm = True
                            sql_source = "llm"
                        else:
                            used_llm = False
                            sql_source = "rules"
                            logger.warning("LLM SQL generation failed; falling back to rule-based pipeline")
                            sql = self._generate_sql_pipeline(nlq, table_name)

                    except Exception as llm_error:
                        used_llm = False
                        sql_source = "rules"
                        error_type = type(llm_error).__name__
                        logger.error(f"LLM generation failed with {error_type}: {llm_error}")
                        sql = self._generate_sql_pipeline(nlq, table_name)
                else:
                    # Legacy pipeline (rule-based first)
                    sql = self._generate_sql_pipeline(nlq, table_name)
                    sql_source = "rules"

            # Post-processing pipeline
            sql = self._post_process_sql(sql, nlq, table_name)

            # Check result cache
            cached_result = self.cache.get_cached_result(sql)
            if cached_result:
                self.metrics.cache_hits += 1
                cached_result.data = self._sanitize_dataframe(cached_result.data)
                return cached_result

            # Execute with repair; on failure optionally fallback to rule-based
            try:
                result = self._execute_with_repair(sql, nlq, table_name)
            except Exception as exec_err:
                if getattr(config.system, "fallback_on_exec_error", True) and sql_source != "rules":
                    fallback_sql = self._build_rule_based_fallback(nlq, table_name)
                    if fallback_sql and fallback_sql.strip() and fallback_sql.strip() != sql.strip():
                        fallback_sql = self._post_process_sql(fallback_sql, nlq, table_name)
                        result = self._execute_with_repair(fallback_sql, nlq, table_name)
                        sql = fallback_sql
                        used_llm = False
                        sql_source = "rules"
                    else:
                        raise
                else:
                    raise

            # Generate summary if needed
            if result.row_count > 0:
                summary = self.llm_manager.summarize_result(nlq, result.data, sql)
                result.summary_text = summary
            elif getattr(config.system, "fallback_on_empty_result", True) and sql_source == "llm":
                # Empty results from LLM SQL: try rule-based fallback once
                fallback_sql = self._build_rule_based_fallback(nlq, table_name)
                if fallback_sql and fallback_sql.strip() and fallback_sql.strip() != sql.strip():
                    fallback_sql = self._post_process_sql(fallback_sql, nlq, table_name)
                    alt_result = self._execute_with_repair(fallback_sql, nlq, table_name)
                    # Prefer non-empty fallback results
                    if alt_result.row_count > 0:
                        result = alt_result
                        sql = fallback_sql
                        result.summary_text = self.llm_manager.summarize_result(nlq, result.data, sql)
                        used_llm = False
                        sql_source = "rules"

            # Sanitize and cache
            result.data = self._sanitize_dataframe(result.data)

            # History and metrics
            self._record_query_history(nlq, sql, result)
            self._update_metrics(result, start_time)

            # Cache final SQL used for this NLQ (post-fallback if any)
            try:
                if sql and isinstance(sql, str):
                    self.cache.cache_sql(nlq, sql)
            except Exception:
                pass

            # Cache result
            if not result.data.isnull().any().any():
                self.cache.cache_result(sql, result)

            return result

        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"Query failed: {e}")
            raise

    def _generate_sql_pipeline(self, nlq: str, table_name: str) -> str:
        """Generate SQL through intent extraction and generation pipeline."""
        # Extract intent
        table_cols = [c.lower() for c in self.db_manager.table_schemas.get(table_name, {}).get("columns", [])]
        intent = self.intent_extractor.extract_intent_from_nlq(nlq, table_cols, table_name, self.db_manager)

        if not intent:
            # Fallback to rule-based generation
            logger.info("LLM generation failed, trying rule-based fallback")
            sql = self._build_rule_based_fallback(nlq, table_name)
        else:
            # Generate SQL from intent
            sql = self.sql_generator.build_sql_from_intent(intent, table_name)

        if not sql:
            raise ValueError("All SQL generation methods failed")

        # Cache generated SQL
        self.cache.cache_sql(nlq, sql)

        return sql

    def _post_process_sql(self, sql: str, nlq: str, table_name: str) -> str:
        """Apply post-processing to generated SQL."""
        # Basic SQL normalization
        sql = self._normalize_sql_table_name(sql, table_name)
        sql = self._quote_columns_with_spaces(sql, table_name)
        sql = self._apply_domain_semantics(nlq, sql, table_name)

        return sql

    def _execute_with_repair(self, sql: str, nlq: str, table_name: str) -> QueryResult:
        """Execute SQL with automatic repair on errors."""
        try:
            with memory_monitor.monitor_operation("Query execution"):
                return self.db_manager.execute_query(sql)
        except Exception as exec_err:
            if self._is_repairable_error(exec_err):
                repaired_sql = self._repair_sql_errors(sql, table_name)
                if repaired_sql and repaired_sql != sql:
                    with memory_monitor.monitor_operation("Query execution (repaired)"):
                        return self.db_manager.execute_query(repaired_sql)
            raise RuntimeError(f"Query execution failed: {sql}. Error: {exec_err}")

    def _is_repairable_error(self, error: Exception) -> bool:
        """Check if error is repairable."""
        text = str(error).lower()
        return (
            "must appear in the group by clause" in text or
            "must be part of an aggregate function" in text or
            "does not have a column named" in text or
            "no such column" in text
        )

    def _repair_sql_errors(self, sql: str, table_name: str) -> Optional[str]:
        """Apply basic repair strategies."""
        # Simple aggregation repair
        measures = ['sales', 'mro', 'mto', 'target', 'primary sales']
        for measure in measures:
            if f"SELECT {measure}" in sql.upper() and "GROUP BY" in sql.upper():
                sql = re.sub(rf"(?i)\b{measure}\b", f"SUM({measure})", sql)
                return sql
        return None

    def _build_rule_based_fallback(self, nlq: str, table_name: str) -> Optional[str]:
        """Basic rule-based SQL generation fallback."""
        try:
            table_cols = [c.lower() for c in self.db_manager.table_schemas.get(table_name, {}).get("columns", [])]
            intent = self.intent_extractor.extract_intent_from_nlq(nlq, table_cols, table_name, self.db_manager)
            if intent:
                return self.sql_generator.build_sql_from_intent(intent, table_name)
            return None
        except Exception as e:
            logger.warning(f"Rule-based fallback failed: {e}")
            return None

    # ---------------- Utility Methods -----------------
    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize DataFrame for JSON serialization."""
        try:
            # Replace NaN/Inf with None
            df = df.replace([np.inf, -np.inf, np.nan], None)
            return df
        except Exception as e:
            logger.warning(f"DataFrame sanitization failed: {e}")
            return df

    def _record_query_history(self, nlq: str, sql: str, result: QueryResult) -> None:
        """Record query in history."""
        try:
            from helper.history import history_manager
            history_manager.record(
                nlq=nlq,
                sql=sql,
                summary=result.summary_text,
                row_count=result.row_count,
                execution_time=result.execution_time,
            )
        except Exception:
            pass  # History recording is optional

    def _update_metrics(self, result: QueryResult, start_time: float) -> None:
        """Update performance metrics."""
        self.metrics.query_count += 1
        self.metrics.total_execution_time += result.execution_time
        self.metrics.cache_misses += 1
        self.metrics.memory_peak_mb = max(self.metrics.memory_peak_mb, result.memory_usage_mb)

        total_time = time.time() - start_time

    def _normalize_sql_table_name(self, sql: str, expected_table: str) -> str:
        """Ensure SQL references the correct table name."""
        try:
            # Clean up unwanted tokens
            sql = re.sub(r"(?i)(assistant|<\|assistant\|>|</s>)$", "", sql).strip()

            # Replace table references
            pattern = r"(?i)\b(FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_\.\"]*)(\s|$)"

            def replace_match(match):
                keyword = match.group(1)
                table_identifier = match.group(2)
                trailing = match.group(3)

                # Strip quotes and schema qualifier
                table_core = table_identifier.strip('`"').split('.')[-1]

                if table_core != expected_table:
                    return f"{keyword} {expected_table}{trailing}"
                return match.group(0)

            return re.sub(pattern, replace_match, sql)

        except Exception as e:
            logger.warning(f"Table name normalization failed: {e}")
            return sql

    def _quote_columns_with_spaces(self, sql: str, table_name: str) -> str:
        """Quote column names containing spaces."""
        try:
            table_info = self.db_manager.table_schemas.get(table_name, {})
            columns = table_info.get("columns", [])
            cols_needing_quotes = [
                col for col in columns
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col)
            ]

            if not cols_needing_quotes:
                return sql

            def replace_outside_quotes(text: str, target: str, replacement: str) -> str:
                result_chars = []
                i = 0
                in_single = in_double = False
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

        except Exception as e:
            logger.warning(f"Column quoting failed: {e}")
            return sql

    def _apply_domain_semantics(self, nlq: str, sql: str, table_name: str) -> str:
        """Apply domain-specific SQL transformations."""
        # Prefer MTO for missed target semantics
        sql = re.sub(r"(?i)\btarget\s*-\s*sales\b", "mto", sql)
        sql = re.sub(r"(?i)\btarget\s*-\s*mro\b", "mto", sql)

        # Normalize boolean predicates
        sql = re.sub(r"(?i)\b(productivity|stockout|assortment)\s*=\s*1\b", r"\1 = TRUE", sql)
        sql = re.sub(r"(?i)\b(productivity|stockout|assortment)\s*=\s*0\b", r"\1 = FALSE", sql)

        return sql

    def cancel_running_query(self) -> bool:
        """Cancel currently running query."""
        return self.db_manager.cancel_query_for_thread()

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cache_hit_rate = (self.cache.cache_stats["hits"] /
                         max(1, self.cache.cache_stats["hits"] + self.cache.cache_stats["misses"]))

        avg_execution_time = (self.metrics.total_execution_time /
                             max(1, self.metrics.query_count))

        # Get LLM failure statistics
        llm_stats = self.llm_manager.get_failure_stats()

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
            },
            "llm_metrics": {
                "llm_success_rate": llm_stats["success_rate_percent"],
                "llm_total_attempts": llm_stats["total_attempts"],
                "llm_failures": llm_stats["total_failures"],
                "llm_failure_breakdown": llm_stats["failure_breakdown"],
                "last_llm_failure": llm_stats["last_failure"]["reason"] if llm_stats["last_failure"]["reason"] else None
            },
            "loaded_tables": {
                name: {"rows": info["total_rows"]}
                for name, info in self.loaded_tables.items()
            }
        }

    def get_llm_health_status(self) -> Dict[str, Any]:
        """Get detailed LLM health and failure analysis."""
        llm_stats = self.llm_manager.get_failure_stats()

        # Analyze failure patterns
        failure_analysis = {}
        if llm_stats["total_attempts"] > 0:
            for failure_type, count in llm_stats["failures"].items():
                if count > 0:
                    percentage = (count / llm_stats["total_attempts"]) * 100
                    failure_analysis[failure_type] = {
                        "count": count,
                        "percentage": round(percentage, 2)
                    }

        return {
            "llm_ready": self.llm_manager.is_ready(),
            "model_info": self.llm_manager.get_model_info(),
            "performance": {
                "success_rate": llm_stats["success_rate_percent"],
                "total_attempts": llm_stats["total_attempts"],
                "successful": llm_stats["successful_generations"],
                "failed": llm_stats["total_failures"]
            },
            "failure_analysis": failure_analysis,
            "last_failure": llm_stats["last_failure"],
            "recommendations": self._get_llm_troubleshooting_recommendations(llm_stats)
        }

    def _get_llm_troubleshooting_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate troubleshooting recommendations based on failure patterns."""
        recommendations = []

        if stats["success_rate_percent"] < 50:
            recommendations.append("⚠️ Low success rate - consider adjusting model parameters")

        failure_breakdown = stats["failure_breakdown"]

        if failure_breakdown["memory_error"] > failure_breakdown["context_window"]:
            recommendations.append("💾 Memory errors dominant - consider reducing context window or freeing RAM")

        if failure_breakdown["context_window"] > 0:
            recommendations.append("📏 Context window issues - try reducing prompt complexity or increasing n_ctx")

        if failure_breakdown["model_error"] > failure_breakdown["validation_error"]:
            recommendations.append("🔧 Model errors - check model compatibility and file integrity")

        if failure_breakdown["empty_response"] > 0:
            recommendations.append("📝 Empty responses - model may need temperature adjustment or prompt refinement")

        if len(recommendations) == 0:
            recommendations.append("✅ LLM performance looks good!")

        return recommendations
