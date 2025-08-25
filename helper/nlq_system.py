import time
import re
import logging
from typing import List, Optional, Tuple, Dict, Any

from config import config
from helper.db import DatabaseManager
from helper.llm_manager import LLMManager
from helper.cache import QueryCache
from helper.types import QueryResult, PerformanceMetrics
from helper.memory import memory_monitor

logger = logging.getLogger(__name__)


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
                    # Rule-based fallback for robustness
                    logger.info("LLM failed to generate SQL, trying rule-based fallback")
                    sql = self._build_rule_based_sql(nlq, table_name)
                    
                if not sql:
                    # Ultimate fallback: generate a safe, simple query
                    logger.info("Rule-based fallback failed, using ultimate fallback")
                    sql = self._build_ultimate_fallback_sql(nlq, table_name)
                    
                if not sql:
                    raise ValueError("All SQL generation methods failed. Please rephrase your question.")
                
                # Cache the generated SQL
                self.cache.cache_sql(nlq, sql)
            
            # Normalize/ensure the correct table name is used in the SQL
            sql = self._normalize_sql_table_name(sql, table_name)
            # Quote any column identifiers that require quoting (e.g., contain spaces)
            sql = self._quote_columns_with_spaces_in_sql(sql, table_name)
            # Enforce semantic column preferences (e.g., prefer "primary sales" when requested)
            sql = self._enforce_semantic_column_preference(nlq, sql, table_name)
            
            # Apply dynamic time logic (months/years) before execution
            sql = self._apply_dynamic_time_logic(nlq, sql, table_name)

            # Enforce domain-specific semantics (mto over target-sales, SUM for totals, boolean normalization, percentages)
            sql = self._enforce_domain_semantics(nlq, sql, table_name)

            # Rewrite window functions to DuckDB-compatible syntax
            sql = self._rewrite_window_functions_to_duckdb(sql)

            # Remove unintended region filters if not explicitly mentioned
            sql = self._strip_unintended_region_filter(nlq, sql)

            # Check result cache
            cached_result = self.cache.get_cached_result(sql)
            if cached_result:
                logger.info("Using cached result")
                self.metrics.cache_hits += 1
                return cached_result
            
            # Execute query with smart repair fallback on aggregation errors
            sql_to_run = sql
            try:
                with memory_monitor.monitor_operation("Query execution"):
                    result = self.db_manager.execute_query(sql_to_run)
            except Exception as exec_err:
                # Try a single smart repair for common binder errors
                if self._is_binder_aggregation_error(exec_err) or self._is_column_binding_error(exec_err):
                    # First fix alias misuse (e.g., s.total_target -> s.target / total_target)
                    repaired_sql = self._repair_alias_misuse_in_sql(sql_to_run)
                    # Then ensure measures are aggregated when grouping
                    repaired_sql = self._repair_aggregation_in_sql(repaired_sql, table_name)
                    if repaired_sql and repaired_sql != sql_to_run:
                        logger.info("Retrying query after SQL repair")
                        sql_to_run = repaired_sql
                        with memory_monitor.monitor_operation("Query execution (repaired)"):
                            result = self.db_manager.execute_query(sql_to_run)
                    else:
                        raise
                else:
                    # Surface the failing SQL for easier debugging
                    raise RuntimeError(f"Query execution failed. Generated SQL: {sql_to_run}. Error: {exec_err}")
            
            # Generate a human-readable summary using the summarizer model
            try:
                summary_text = self.llm_manager.summarize_result(nlq, result.data, sql_to_run)
                result.summary_text = summary_text
            except Exception:
                result.summary_text = ""

            # Persist history (NLQ, SQL, summary and metadata)
            try:
                from helper.history import history_manager
                history_manager.record(
                    nlq=nlq,
                    sql=sql_to_run,
                    summary=result.summary_text,
                    row_count=result.row_count,
                    execution_time=result.execution_time,
                )
            except Exception:
                pass

            # Cache the result (for the actual SQL executed)
            self.cache.cache_result(sql_to_run, result)
            
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

    def cancel_running_query(self) -> bool:
        """Cancel the query running in the current thread if possible."""
        return self.db_manager.cancel_query_for_thread()

    # ---------------- Smart SQL repair utilities ---------------------------
    def _is_binder_aggregation_error(self, err: Exception) -> bool:
        text = str(err).lower()
        return (
            "must appear in the group by clause" in text or
            "must be part of an aggregate function" in text
        )

    def _is_column_binding_error(self, err: Exception) -> bool:
        text = str(err).lower()
        return (
            "does not have a column named" in text or
            "no such column" in text or
            "not found in" in text
        )

    def _repair_aggregation_in_sql(self, sql: str, table_name: str) -> str:
        """Attempt to wrap measure columns in SUM() within SELECT/ORDER BY.
        This addresses errors when non-aggregated measures are selected alongside grouped dimensions.
        Keeps dimensional columns untouched. Best-effort regex approach.
        """
        try:
            # Measures to aggregate
            measures = [
                'sales', 'mro', 'mto', 'target', 'primary sales',
                'stockout_mro', 'unproductive_mro', 'unassorted_mro'
            ]

            # Build patterns for alias-qualified and bare names
            # e.g., s.sales -> SUM(s.sales); "primary sales" -> SUM("primary sales")
            # Operate only inside SELECT ... FROM and ORDER BY ... (until LIMIT/end)
            def find_section(text: str, start_kw: str, end_kws: List[str]) -> Tuple[int, int]:
                m = re.search(rf"(?is)\b{start_kw}\b", text)
                if not m:
                    return (-1, -1)
                start = m.end()
                end = len(text)
                for ek in end_kws:
                    em = re.search(rf"(?is)\b{ek}\b", text[start:])
                    if em:
                        end = start + em.start()
                        break
                return (start, end)

            def wrap_measures(segment: str) -> str:
                if not segment:
                    return segment
                # Replace alias-qualified measures first
                for meas in sorted(measures, key=len, reverse=True):
                    quoted = f'"{meas}"'
                    # alias."primary sales" or alias.sales
                    pattern_alias_quoted = rf"(?<!SUM\()\b([A-Za-z_][A-Za-z0-9_]*)\.(\"{re.escape(meas)}\")\b"
                    pattern_alias_bare = rf"(?<!SUM\()\b([A-Za-z_][A-Za-z0-9_]*)\.{re.escape(meas)}\b"
                    segment = re.sub(pattern_alias_quoted, r"SUM(\1.\2)", segment, flags=re.IGNORECASE)
                    segment = re.sub(pattern_alias_bare, rf"SUM(\1.{meas})", segment, flags=re.IGNORECASE)
                    # bare quoted or bare name
                    pattern_bare_quoted = rf"(?<!SUM\()\b(\"{re.escape(meas)}\")\b"
                    pattern_bare = rf"(?<!SUM\()\b{re.escape(meas)}\b"
                    segment = re.sub(pattern_bare_quoted, r"SUM(\1)", segment, flags=re.IGNORECASE)
                    segment = re.sub(pattern_bare, rf"SUM({meas})", segment, flags=re.IGNORECASE)
                return segment

            lower_sql = sql.lower()
            # SELECT segment
            s_start = re.search(r"(?is)\bselect\b", sql)
            f_from = re.search(r"(?is)\bfrom\b", sql)
            if s_start and f_from and s_start.end() < f_from.start():
                pre = sql[:s_start.end()]
                select_body = sql[s_start.end():f_from.start()]
                post = sql[f_from.start():]
                select_body = wrap_measures(select_body)
                sql = pre + select_body + post

            # ORDER BY segment
            ob_start, ob_end = find_section(sql, 'order\s+by', ['limit'])
            if ob_start != -1:
                pre = sql[:ob_start]
                ob_body = sql[ob_start:ob_end]
                post = sql[ob_end:]
                ob_body = wrap_measures(ob_body)
                sql = pre + ob_body + post

            return sql
        except Exception as e:
            logger.warning(f"_repair_aggregation_in_sql failed: {e}")
            return sql

    def _repair_alias_misuse_in_sql(self, sql: str) -> str:
        """Repair misuse of SELECT aliases as if they were base columns (e.g., s.total_target).
        Replace patterns like SUM(s.total_target) -> SUM(s.target), SUM(total_sales) -> SUM(sales),
        and standalone s.total_target -> total_target where appropriate.
        """
        try:
            updated = sql
            # Inside SUM()
            updated = re.sub(r"(?is)SUM\(\s*([A-Za-z_][A-Za-z0-9_]*)\.total_target\s*\)", r"SUM(\1.target)", updated)
            updated = re.sub(r"(?is)SUM\(\s*([A-Za-z_][A-Za-z0-9_]*)\.total_sales\s*\)", r"SUM(\1.sales)", updated)
            updated = re.sub(r"(?is)SUM\(\s*total_target\s*\)", "SUM(target)", updated)
            updated = re.sub(r"(?is)SUM\(\s*total_sales\s*\)", "SUM(sales)", updated)

            # Standalone alias-qualified references -> plain alias
            updated = re.sub(r"(?is)\b([A-Za-z_][A-Za-z0-9_]*)\.total_target\b", "total_target", updated)
            updated = re.sub(r"(?is)\b([A-Za-z_][A-Za-z0-9_]*)\.total_sales\b", "total_sales", updated)
            return updated
        except Exception as e:
            logger.warning(f"_repair_alias_misuse_in_sql failed: {e}")
            return sql

    # ---------------- Rule-based fallback SQL ------------------------------
    def _build_rule_based_sql(self, nlq: str, table_name: str) -> Optional[str]:
        """Construct a simple, robust SQL for common query shapes when LLM fails.
        Supports queries like:
          - Which <dimension> has most/least <metric or opportunity> in <month/quarter/year>?
          - Which <dimension> has most potential to grow in <month>?
        Metrics recognized: sales, mro (growth opportunity/potential improvement), mto, target, primary sales.
        Dimensions recognized: region, city, area, territory, distributor, route, brand, sku, customer.
        """
        try:
            nlq_lower = nlq.lower()
            # Normalize common plural forms in NLQ to match column names
            plural_map = {
                "territories": "territory",
                "cities": "city",
                "areas": "area",
                "regions": "region",
                "routes": "route",
                "brands": "brand",
                "customers": "customer",
                "distributors": "distributor",
                "skus": "sku",
                "targets": "target",
                "sales": "sales",
            }
            nlq_norm = nlq_lower
            for plural, singular in plural_map.items():
                nlq_norm = re.sub(rf"\b{plural}\b", singular, nlq_norm)
            
            # Check if this is a growth query
            is_growth_query = self._is_growth_query(nlq_norm)
            
            # Determine dimension
            candidate_dims = [
                "region", "city", "area", "territory", "distributor", "route", "brand", "sku", "customer"
            ]
            table_cols = [c.lower() for c in self.db_manager.table_schemas.get(table_name, {}).get("columns", [])]
            
            if not table_cols:
                logger.warning("No table columns found for rule-based SQL generation")
                return None
            
            chosen_dim = None
            for dim in candidate_dims:
                if dim in nlq_norm and dim in table_cols:
                    chosen_dim = dim
                    break
            if chosen_dim is None:
                # Default to region if present, else first textual column
                if "region" in table_cols:
                    chosen_dim = "region"
                else:
                    chosen_dim = next((c for c in table_cols if c not in {"month","year"}), None)
            if chosen_dim is None:
                logger.warning("Could not determine dimension for rule-based SQL")
                return None

            # Determine metric
            metric = self._choose_metric_from_nlq(nlq_norm, table_cols)
            if metric is None:
                metric = "sales" if "sales" in table_cols else ("mro" if "mro" in table_cols else None)
            if metric is None:
                logger.warning("Could not determine metric for rule-based SQL")
                return None

            # Direction (most vs least)
            asc_terms = ["least", "lowest", "worst", "minimum", "min"]
            desc_terms = ["most", "highest", "top", "best", "maximum", "max"]
            order_dir = "DESC"
            if any(t in nlq_norm for t in asc_terms):
                order_dir = "ASC"
            elif any(t in nlq_norm for t in desc_terms):
                order_dir = "DESC"

            # Determine LIMIT N for top/bottom queries
            limit_n = 1
            m_topn = re.search(r"\b(top|bottom)\s*(\d+)\b", nlq_norm)
            if m_topn:
                try:
                    limit_n = max(1, int(m_topn.group(2)))
                except Exception:
                    limit_n = 5
            elif "top" in nlq_norm or "bottom" in nlq_norm:
                limit_n = 5

            # Time filters (month/year)
            year, months = self._extract_periods_from_nlq(nlq_lower)
            if months and year is None:
                # Use latest year containing the latest mentioned month
                try:
                    guess_year = self.db_manager.get_latest_year_for_month(table_name, max(months))
                    year = guess_year
                except Exception:
                    logger.warning("Could not determine year for month, using latest available")
                    periods = self.db_manager.get_latest_periods(table_name, 1)
                    if periods:
                        year = periods[0][0]
            if year is None and not months:
                # Default to latest month
                try:
                    periods = self.db_manager.get_latest_periods(table_name, 1)
                    if periods:
                        year, m = periods[0]
                        months = [m]
                except Exception:
                    logger.warning("Could not get latest periods, using defaults")
                    year = 2024
                    months = [12]

            # Quote identifiers
            dim_sql = self.db_manager._quote_identifier(chosen_dim)
            metric_sql = self.db_manager._quote_identifier(metric)

            alias_metric = re.sub(r"[^A-Za-z0-9_]+", "_", metric)
            
            # If this is a growth query, build growth-specific SQL
            if is_growth_query:
                growth_sql = self._build_growth_sql(nlq_norm, table_name, chosen_dim, metric, year, months, limit_n, order_dir)
                if growth_sql:
                    return growth_sql
                # If growth SQL fails, fall back to simple aggregation
            
            # Standard aggregation query
            where_parts = []
            if year:
                where_parts.append(f"year = {int(year)}")
            if months:
                month_list = ", ".join(str(int(m)) for m in sorted(set(months)))
                where_parts.append(f"month IN ({month_list})")
            where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

            sql = (
                f"SELECT {dim_sql}, SUM({metric_sql}) AS total_{alias_metric} "
                f"FROM {table_name}"
                f"{where_clause} "
                f"GROUP BY {dim_sql} "
                f"ORDER BY total_{alias_metric} {order_dir} "
                f"LIMIT {limit_n};"
            )
            
            logger.info(f"Generated rule-based SQL: {sql}")
            return sql
            
        except Exception as e:
            logger.warning(f"_build_rule_based_sql failed: {e}")
            return None

    def _build_growth_sql(self, nlq_lower: str, table_name: str, dimension: str, metric: str, 
                          year: Optional[int], months: List[int], limit_n: int, order_dir: str) -> str:
        """Build growth analysis SQL using DuckDB-compatible syntax without window functions."""
        try:
            # Determine the months to compare
            if months and len(months) >= 2:
                # User specified multiple months, use them
                month1, month2 = sorted(months)[-2:]  # Last two months
            elif months and len(months) == 1:
                # User specified one month, compare with previous month
                month1 = months[0]
                month2 = month1 - 1 if month1 > 1 else 12
                if month1 == 1:
                    year = year - 1 if year else None
            else:
                # No months specified, use latest two available
                periods = self.db_manager.get_latest_periods(table_name, limit=2)
                if len(periods) >= 2:
                    year, month1 = periods[0]
                    _, month2 = periods[1]
                else:
                    # Only one period available, use it and previous month
                    year, month1 = periods[0]
                    month2 = month1 - 1 if month1 > 1 else 12
                    if month1 == 1:
                        year = year - 1

            # Ensure we have valid months
            if month1 is None or month2 is None:
                month1, month2 = 12, 11  # Default to December vs November
            
            # Ensure we have a valid year
            if year is None:
                # Get the latest available year for the mentioned month
                try:
                    if months and len(months) >= 1:
                        # Get the latest year for the first mentioned month
                        latest_year = self.db_manager.get_latest_year_for_month(table_name, months[0])
                        if latest_year:
                            year = latest_year
                        else:
                            # Fallback: get the latest year from any available periods
                            periods = self.db_manager.get_latest_periods(table_name, limit=1)
                            if periods:
                                year = periods[0][0]
                            else:
                                year = 2025  # Last resort default
                    else:
                        # No months specified, get the latest year from available periods
                        periods = self.db_manager.get_latest_periods(table_name, limit=1)
                        if periods:
                            year = periods[0][0]
                        else:
                            year = 2025  # Last resort default
                except Exception as e:
                    logger.warning(f"Failed to get latest year for month: {e}")
                    # Fallback: get the latest year from any available periods
                    try:
                        periods = self.db_manager.get_latest_periods(table_name, limit=1)
                        if periods:
                            year = periods[0][0]
                        else:
                            year = 2025  # Last resort default
                    except Exception:
                        year = 2025  # Last resort default

            # Quote identifiers
            dim_sql = self.db_manager._quote_identifier(dimension)
            metric_sql = self.db_manager._quote_identifier(metric)

            # Build the growth SQL
            sql = f"""SELECT {dim_sql},
                    SUM(CASE WHEN month = {month1} THEN {metric_sql} ELSE 0 END) AS {metric}_month_{month1},
                    SUM(CASE WHEN month = {month2} THEN {metric_sql} ELSE 0 END) AS {metric}_month_{month2},
                    ((SUM(CASE WHEN month = {month1} THEN {metric_sql} ELSE 0 END) -
                    SUM(CASE WHEN month = {month2} THEN {metric_sql} ELSE 0 END)) /
                    NULLIF(SUM(CASE WHEN month = {month2} THEN {metric_sql} ELSE 0 END), 0)) * 100 AS growth_percentage
                    FROM {table_name}
                    WHERE year = {year} AND month IN ({month2}, {month1})
                    GROUP BY {dim_sql}
                    HAVING SUM(CASE WHEN month = {month2} THEN {metric_sql} ELSE 0 END) > 0
                    ORDER BY growth_percentage {order_dir}
                    LIMIT {limit_n};"""

            return sql
        except Exception as e:
            logger.warning(f"_build_growth_sql failed: {e}")
            # Fallback to simple aggregation
            return self._build_simple_growth_sql(table_name, dimension, metric, year, months, limit_n, order_dir)

    def _build_simple_growth_sql(self, table_name: str, dimension: str, metric: str, 
                                year: Optional[int], months: List[int], limit_n: int, order_dir: str) -> str:
        """Fallback simple growth SQL if complex growth analysis fails."""
        try:
            # Use latest two months available
            periods = self.db_manager.get_latest_periods(table_name, limit=2)
            if len(periods) >= 2:
                year, month1 = periods[0]
                _, month2 = periods[1]
            else:
                year, month1 = periods[0]
                month2 = month1 - 1 if month1 > 1 else 12

            dim_sql = self.db_manager._quote_identifier(dimension)
            metric_sql = self.db_manager._quote_identifier(metric)

            sql = f"""SELECT {dim_sql}, SUM({metric_sql}) AS total_{metric}
                    FROM {table_name}
                    WHERE year = {year} AND month IN ({month2}, {month1})
                    GROUP BY {dim_sql}
                    ORDER BY total_{metric} {order_dir}
                    LIMIT {limit_n};"""

            return sql
        except Exception as e:
            logger.warning(f"_build_simple_growth_sql failed: {e}")
            return None

    def _choose_metric_from_nlq(self, nlq_lower: str, table_cols: List[str]) -> Optional[str]:
        """Infer metric column from NLQ text with synonyms and fallbacks."""
        try:
            if any(term in nlq_lower for term in [
                "growth opportunity", "growth potential", "potential to grow", "potential",
                "potential improvement", "missed revenue", "missed gain",
                "untapped potential", "lost earnings", "mro", "opportunity"
            ]):
                if "mro" in table_cols:
                    return "mro"
            if any(term in nlq_lower for term in ["missed target", "target missed", "sales shortfall", "performance gap", "mto"]):
                if "mto" in table_cols:
                    return "mto"
            if any(term in nlq_lower for term in ["primary sales", "primary contribution"]):
                if "primary sales" in table_cols:
                    return "primary sales"
            if any(term in nlq_lower for term in ["target"]):
                if "target" in table_cols:
                    return "target"
            if any(term in nlq_lower for term in ["sales", "revenue", "contribution"]):
                if "sales" in table_cols:
                    return "sales"
            return None
        except Exception:
            return None

    # ---------------- Dynamic time and growth handling ---------------------
    def _apply_dynamic_time_logic(self, nlq: str, sql: str, table_name: str) -> str:
        """Inject sensible default time filters and rewrite growth queries to use
        latest available periods when the user has not explicitly specified months/years.
        """
        try:
            # Only act if table has month/year columns
            if not self.db_manager.has_columns(table_name, ["month", "year"]):
                return sql

            nlq_lower = nlq.lower()
            extracted_year, extracted_months = self._extract_periods_from_nlq(nlq_lower)

            # Normalize invalid year placeholders like 20XX/XXXX or short '20' as early as possible
            sql = self._normalize_placeholder_years(sql, table_name, extracted_months)

            # If SQL already has some time filters, optionally strengthen with missing pieces
            if self._sql_has_time_filters(sql):
                # If months are present in SQL but year is missing and NLQ specified a year, inject the year
                if self._sql_has_month_filter(sql) and not self._sql_has_year_filter(sql) and extracted_year is not None:
                    return self._inject_time_filter_into_sql(sql, year=extracted_year, months=[])
                # If NLQ asked for growth without explicit months in NLQ or SQL, rewrite to latest
                if self._is_growth_query(nlq_lower) and not extracted_months:
                    return self._rewrite_growth_to_latest(sql, table_name)
                return sql

            # No time filters present in SQL
            # Respect explicit periods in NLQ if we detected them
            if extracted_months or extracted_year is not None:
                # Try to fill missing year using latest available if not specified
                year_to_use: Optional[int] = extracted_year
                if year_to_use is None:
                    if len(extracted_months) >= 1:
                        # Find the latest year that contains the latest mentioned month
                        month_latest = max(extracted_months) if extracted_months else None
                        if month_latest is not None:
                            guess_year = self.db_manager.get_latest_year_for_month(table_name, month_latest)
                            year_to_use = guess_year
                # If still None, fall back to latest
                if year_to_use is None:
                    latest_periods = self.db_manager.get_latest_periods(table_name, limit=1)
                    if latest_periods:
                        year_to_use = latest_periods[0][0]

                if self._is_growth_query(nlq_lower):
                    # If only one month given (e.g., December), use that as latest and include previous month
                    month_latest = max(extracted_months) if extracted_months else None
                    if month_latest is not None and year_to_use is not None:
                        return self._rewrite_growth_to_specific(sql, table_name, year_to_use, month_latest)
                    # Fallback to latest two if month not identifiable
                    return self._rewrite_growth_to_latest(sql, table_name)

                # Non-growth queries: inject specified months (or single month) with chosen year
                return self._inject_time_filter_into_sql(sql, year=year_to_use if year_to_use else 0, months=sorted(extracted_months))

            # Growth question with no explicit months: use latest two
            if self._is_growth_query(nlq_lower):
                return self._rewrite_growth_to_latest(sql, table_name)

            # Non-growth and no explicit period: constrain to latest month by default
            latest_periods = self.db_manager.get_latest_periods(table_name, limit=1)
            if latest_periods:
                latest_year, latest_month = latest_periods[0]
                return self._inject_time_filter_into_sql(sql, year=latest_year, months=[latest_month])
            return sql
        except Exception as e:
            logger.warning(f"_apply_dynamic_time_logic failed: {e}")
            return sql

    def _normalize_placeholder_years(self, sql: str, table_name: str, months_hint: List[int]) -> str:
        """Replace placeholder years like 20XX/XXXX or partial years (20) in SQL with
        actual 4-digit years present in the data, preferring the latest year that contains
        the referenced month(s). If no month is referenced, use the latest year overall.
        """
        try:
            # Detect presence of placeholders
            placeholder_patterns = [r"\b20XX\b", r"\bXXXX\b", r"\b20\b"]
            if not any(re.search(pat, sql, flags=re.IGNORECASE) for pat in placeholder_patterns):
                return sql

            latest_periods = self.db_manager.get_latest_periods(table_name, limit=1)
            latest_year = latest_periods[0][0] if latest_periods else None

            # If we have a month hint, try to get latest year for that month
            year_for_month = None
            if months_hint:
                try:
                    year_for_month = self.db_manager.get_latest_year_for_month(table_name, max(months_hint))
                except Exception:
                    year_for_month = None

            chosen_year = year_for_month or latest_year
            if chosen_year is None:
                return sql

            updated = sql
            updated = re.sub(r"(?is)\b20XX\b", str(chosen_year), updated)
            updated = re.sub(r"(?is)\bXXXX\b", str(chosen_year), updated)
            # Replace bare 'year = 20' patterns only (avoid accidental number replacements)
            updated = re.sub(r"(?is)(year\s*=\s*)20\b", rf"\g<1>{chosen_year}", updated)
            return updated
        except Exception:
            return sql

    # ---------------- Domain-specific SQL normalization --------------------
    def _enforce_domain_semantics(self, nlq: str, sql: str, table_name: str) -> str:
        """Apply domain-specific fixes:
        - Prefer mto for 'missed target' semantics; avoid target - sales or target - mro
        - Use SUM() for totals of mro/mto/sales/target when grouping
        - Normalize boolean predicates to = 1 / = 0
        - Ensure region filters from NLQ are present in SQL
        - Add percentage calculations for stockout/productivity when NLQ asks for percent
        """
        try:
            updated = sql
            table_cols = [c.lower() for c in self.db_manager.table_schemas.get(table_name, {}).get("columns", [])]
            nlq_lower = nlq.lower()

            # 1) Normalize TRUE/FALSE to 1/0 for boolean-like predicates
            updated = re.sub(r"(?is)\b=\s*TRUE\b", "= 1", updated)
            updated = re.sub(r"(?is)\b=\s*FALSE\b", "= 0", updated)

            # 2) Prefer mto for missed target semantics
            if any(k in nlq_lower for k in ["missed target", "target missed", "target gap", "gap to target", "mto"]):
                if "mto" in table_cols:
                    # Replace direct arithmetic patterns with mto
                    patterns = [
                        (r"(?is)SUM\s*\(\s*target\s*-\s*sales\s*\)", "SUM(mto)"),
                        (r"(?is)SUM\s*\(\s*\"primary sales\"\s*-\s*target\s*\)", "SUM(mto)"),
                        (r"(?is)SUM\s*\(\s*target\s*-\s*\"primary sales\"\s*\)", "SUM(mto)"),
                        (r"(?is)SUM\s*\(\s*target\s*-\s*mro\s*\)", "SUM(mto)"),
                        (r"(?is)\btarget\s*-\s*sales\b", "mto"),
                        (r"(?is)\btarget\s*-\s*mro\b", "mto"),
                        (r"(?is)SUM\s*\(\s*target\s*\)\s*-\s*SUM\s*\(\s*sales\s*\)", "SUM(mto)"),
                    ]
                    for pat, rep in patterns:
                        updated = re.sub(pat, rep, updated)
                    # Ensure aggregation alias if ordering by a difference term
                    updated = re.sub(r"(?is)ORDER\s+BY\s*\((?:[^\)]*target[^\)]*-[^\)]*sales[^\)]*)\)\s*DESC", "ORDER BY SUM(mto) DESC", updated)

                    # Remove unrelated stockout filter if present (common user mistake)
                    updated = re.sub(r"(?is)\bAND\s+stockout\s*=\s*(1|0)\b", "", updated)
                    updated = re.sub(r"(?is)\bWHERE\s+stockout\s*=\s*(1|0)\b", "WHERE 1=1", updated)

            # 3) Prefer SUM for totals when grouping
            if " group by " in updated.lower():
                for meas in ["mro", "mto", "sales", "target", '"primary sales"']:
                    updated = re.sub(rf"(?is)\bMAX\s*\(\s*{re.escape(meas)}\s*\)", f"SUM({meas})", updated)

            # 4) Remove unintended region filter if NLQ omits region, then ensure region if explicitly mentioned
            updated = self._remove_region_filter_if_not_requested(nlq, updated)
            updated = self._ensure_region_filter(nlq, updated)

            # 5) Percentages for stockout/productivity when requested
            if any(w in nlq_lower for w in ["percent", "percentage", "%", " ratio", " rate"]):
                # Stockout percentage
                if "stockout" in updated.lower():
                    if "stockout_percentage" not in updated.lower():
                        updated = self._ensure_percentage_for_flag(updated, flag_column="stockout", percent_alias="stockout_percentage")
                # Productivity percentage
                if "productivity" in updated.lower():
                    if "productivity_percentage" not in updated.lower():
                        updated = self._ensure_percentage_for_flag(updated, flag_column="productivity", percent_alias="productivity_percentage")

            # 6) Window functions should not be in WHERE: move them to QUALIFY
            if re.search(r"(?is)\bwhere\b.*\bover\s*\(", updated):
                updated = self._rewrite_window_filters_to_qualify(updated)

            return updated
        except Exception as e:
            logger.warning(f"_enforce_domain_semantics failed: {e}")
            return sql

    def _ensure_percentage_for_flag(self, sql: str, flag_column: str, percent_alias: str) -> str:
        """Ensure SELECT computes percentage for a 0/1 flag by customer within groups.
        Adds COUNT(DISTINCT customer) and percentage expression if missing.
        """
        try:
            m_select = re.search(r"(?is)\bselect\b", sql)
            m_from = re.search(r"(?is)\bfrom\b", sql)
            if not m_select or not m_from or m_select.end() >= m_from.start():
                return sql
            head = sql[:m_select.end()]
            select_body = sql[m_select.end():m_from.start()].rstrip()
            tail = sql[m_from.start():]

            # Add total customers if not present
            if re.search(r"(?is)count\s*\(\s*distinct\s*customer\s*\)\s*as\s*total_\w+", select_body) is None:
                select_body += ", COUNT(DISTINCT customer) AS total_customers"

            # Add percentage expression
            percent_expr = (
                f"(COUNT(DISTINCT CASE WHEN {flag_column} = 1 THEN customer END) * 100.0 / "
                f"NULLIF(COUNT(DISTINCT customer), 0)) AS {percent_alias}"
            )
            if percent_alias.lower() not in select_body.lower():
                select_body += f", {percent_expr}"

            updated = head + select_body + tail
            # Prefer ordering by the percentage if ORDER BY references count
            updated = re.sub(r"(?is)order\s+by\s+\w*stockout\w*_?count\w*\s+(asc|desc)", f"ORDER BY {percent_alias} \\1", updated)
            return updated
        except Exception:
            return sql

    def _ensure_region_filter(self, nlq: str, sql: str) -> str:
        """Inject region filter into SQL if NLQ mentions a region and SQL lacks a region filter."""
        try:
            nlq_lower = nlq.lower()
            
            # Enhanced region detection patterns
            # 1. Exact match: Central-A, North B, South-A, etc.
            exact_match = re.search(r"\b(north|south|central|east|west)[\-\s]?([ab])\b", nlq_lower)
            if exact_match:
                region = exact_match.group(1).capitalize() + "-" + exact_match.group(2).upper()
                # If SQL already filters on region, skip
                if re.search(r"(?is)\bregion\s*=\s*'[^']+'", sql):
                    return sql
                # Inject exact region filter
                return self._inject_filter_clause(sql, f"region = '{region}'")
            
            # 2. Partial match: Central, North, South, etc. (without A/B suffix)
            partial_match = re.search(r"\b(north|south|central|east|west)\b", nlq_lower)
            if partial_match:
                region_base = partial_match.group(1).capitalize()
                # If SQL already filters on region, skip
                if re.search(r"(?is)\bregion\s*=\s*'[^']+'", sql):
                    return sql
                # Use LIKE operator for partial matching (DuckDB compatible)
                # This will match Central-A, Central-B, Central-C, etc.
                return self._inject_filter_clause(sql, f"region LIKE '{region_base}-%'")
            
            # 3. Additional patterns for more flexible matching
            # Handle "Central region", "North area", etc.
            region_with_context = re.search(r"\b(north|south|central|east|west)\s+(region|area|territory)\b", nlq_lower)
            if region_with_context:
                region_base = region_with_context.group(1).capitalize()
                if re.search(r"(?is)\bregion\s*=\s*'[^']+'", sql):
                    return sql
                return self._inject_filter_clause(sql, f"region LIKE '{region_base}-%'")
            
            # 4. Handle abbreviated forms: "Central reg", "North reg", etc.
            region_abbrev = re.search(r"\b(north|south|central|east|west)\s+reg\b", nlq_lower)
            if region_abbrev:
                region_base = region_abbrev.group(1).capitalize()
                if re.search(r"(?is)\bregion\s*=\s*'[^']+'", sql):
                    return sql
                return self._inject_filter_clause(sql, f"region LIKE '{region_base}-%'")
            
            return sql
        except Exception:
            return sql

    def _remove_region_filter_if_not_requested(self, nlq: str, sql: str) -> str:
        """Remove region filters when NLQ does not mention a region."""
        try:
            nlq_lower = nlq.lower()
            # If NLQ mentions region, do nothing
            if re.search(r"\b(north|south|central|east|west)[\-\s]?([ab])\b", nlq_lower):
                return sql
            # Also check for partial region mentions
            if re.search(r"\b(north|south|central|east|west)\b", nlq_lower):
                return sql
            # Check for region with context
            if re.search(r"\b(north|south|central|east|west)\s+(region|area|territory)\b", nlq_lower):
                return sql
            # Check for abbreviated forms
            if re.search(r"\b(north|south|central|east|west)\s+reg\b", nlq_lower):
                return sql

            # If no WHERE present, nothing to do
            m_where = re.search(r"(?is)\bwhere\b", sql)
            if not m_where:
                return sql

            where_start = m_where.end()
            # Find end of WHERE clause
            end_tokens = [" GROUP BY ", " HAVING ", " QUALIFY ", " ORDER BY ", " LIMIT ", ";"]
            lower_sql = sql.lower()
            where_end = len(sql)
            for tok in end_tokens:
                pos = lower_sql.find(tok.lower(), where_start)
                if pos != -1:
                    where_end = min(where_end, pos)
            where_body = sql[where_start:where_end]
            if not where_body:
                return sql

            # Split WHERE body by top-level AND without breaking quoted sections
            def split_top_level_and(expr: str) -> List[str]:
                parts: List[str] = []
                buf: List[str] = []
                depth = 0
                in_single = False
                in_double = False
                i = 0
                while i < len(expr):
                    ch = expr[i]
                    if ch == "'" and not in_double:
                        in_single = not in_single
                        buf.append(ch)
                        i += 1
                        continue
                    if ch == '"' and not in_single:
                        in_double = not in_double
                        buf.append(ch)
                        i += 1
                        continue
                    if not in_single and not in_double:
                        if ch == '(':
                            depth += 1
                        elif ch == ')':
                            depth = max(0, depth - 1)
                        if depth == 0 and expr[i:i+4].lower() == ' and' and (i == 0 or expr[i-1].isspace()):
                            parts.append(''.join(buf).strip())
                            buf = []
                            i += 4
                            continue
                    buf.append(ch)
                    i += 1
                last = ''.join(buf).strip()
                if last:
                    parts.append(last)
                return [p for p in parts if p]

            conditions = split_top_level_and(where_body)
            # Identify region conditions (including LIKE patterns)
            region_conds = []
            keep_conds = []
            for c in conditions:
                if (re.search(r"(?is)\bregion\s*=\s*'[^']+'", c) or 
                    re.search(r"(?is)\bregion\s+in\s*\([^)]+\)", c) or
                    re.search(r"(?is)\bregion\s+like\s*'[^']+'", c)):
                    region_conds.append(c)
                else:
                    keep_conds.append(c)

            # Nothing to strip
            if not region_conds:
                return sql

            before_where = sql[:m_where.start()]
            after_where = sql[where_end:]
            if keep_conds:
                new_where = 'WHERE ' + ' AND '.join(keep_conds) + ' '
                return before_where + new_where + after_where
            else:
                # Remove entire WHERE if only region was present
                return before_where + after_where
        except Exception:
            return sql

    def _inject_filter_clause(self, sql: str, clause: str) -> str:
        """General-purpose WHERE/AND injection preserving clause order."""
        try:
            tokens = [" GROUP BY ", " ORDER BY ", " LIMIT "]
            lower = sql.lower()
            insert_pos = len(sql)
            for token in tokens:
                pos = lower.find(token.lower())
                if pos != -1:
                    insert_pos = min(insert_pos, pos)
            head = sql[:insert_pos]
            tail = sql[insert_pos:]
            if re.search(r"(?is)\bwhere\b", head):
                head = re.sub(r"(?is)\bwhere\b", "WHERE", head, count=1)
                head = re.sub(r"(?is)WHERE", f"WHERE {clause} AND", head, count=1)
            else:
                head = head.rstrip("; ") + f" WHERE {clause}"
            return head + tail
        except Exception:
            return sql

    def _rewrite_window_filters_to_qualify(self, sql: str) -> str:
        """Move window function predicates from WHERE to QUALIFY.
        Splits WHERE conditions on top-level ANDs; conditions containing OVER(...) are moved
        into a QUALIFY clause. If QUALIFY exists, they are appended with AND.
        """
        try:
            m_where = re.search(r"(?is)\bwhere\b", sql)
            if not m_where:
                return sql
            where_start = m_where.end()
            # Determine end of WHERE
            end_tokens = [" GROUP BY ", " HAVING ", " QUALIFY ", " ORDER BY ", " LIMIT ", ";"]
            lower_sql = sql.lower()
            where_end = len(sql)
            for tok in end_tokens:
                pos = lower_sql.find(tok.lower(), where_start)
                if pos != -1:
                    where_end = min(where_end, pos)
            where_body = sql[where_start:where_end].strip()
            if not where_body or 'over(' not in where_body.lower():
                return sql

            # Split conditions by top-level AND
            def split_top_level_and(expr: str) -> List[str]:
                parts: List[str] = []
                buf: List[str] = []
                depth = 0
                in_single = False
                in_double = False
                i = 0
                while i < len(expr):
                    ch = expr[i]
                    if ch == "'" and not in_double:
                        in_single = not in_single
                        buf.append(ch)
                        i += 1
                        continue
                    if ch == '"' and not in_single:
                        in_double = not in_double
                        buf.append(ch)
                        i += 1
                        continue
                    if not in_single and not in_double:
                        if ch == '(':
                            depth += 1
                        elif ch == ')':
                            depth = max(0, depth - 1)
                        if depth == 0 and expr[i:i+4].lower() == ' and' and (i == 0 or expr[i-1].isspace()):
                            parts.append(''.join(buf).strip())
                            buf = []
                            i += 4
                            continue
                    buf.append(ch)
                    i += 1
                last = ''.join(buf).strip()
                if last:
                    parts.append(last)
                return [p for p in parts if p]

            conditions = split_top_level_and(where_body)
            window_conds = [c for c in conditions if ('over(' in c.lower()) or re.search(r"(?is)\b(lag|lead|row_number|rank|dense_rank)\s*\(", c)]
            non_window_conds = [c for c in conditions if c not in window_conds]
            if not window_conds:
                return sql

            before_where = sql[:m_where.start()]
            after_where = sql[where_end:]
            new_where = ''
            if non_window_conds:
                new_where = 'WHERE ' + ' AND '.join(non_window_conds) + ' '

            # Insert or extend QUALIFY
            m_qual = re.search(r"(?is)\bqualify\b", after_where)
            if m_qual:
                qual_start = m_qual.end()
                after = after_where[:qual_start] + ' ' + ' AND '.join(window_conds) + after_where[qual_start:]
                return before_where + new_where + after
            else:
                # Place QUALIFY before ORDER BY/LIMIT
                insert_pos = len(after_where)
                for tok in [" ORDER BY ", " LIMIT "]:
                    pos = after_where.lower().find(tok.lower())
                    if pos != -1:
                        insert_pos = min(insert_pos, pos)
                after_head = after_where[:insert_pos]
                after_tail = after_where[insert_pos:]
                qualify_str = 'QUALIFY ' + ' AND '.join(window_conds) + ' '
                return before_where + new_where + after_head + qualify_str + after_tail
        except Exception as e:
            logger.warning(f"_rewrite_window_filters_to_qualify failed: {e}")
            return sql

    def _is_growth_query(self, nlq_lower: str) -> bool:
        return any(word in nlq_lower for word in ["growth", "grow", "growing", "expansion", "increase", "decrease"]) \
            or " vs " in nlq_lower or "compare" in nlq_lower

    def _sql_has_time_filters(self, sql: str) -> bool:
        pat = r"(?is)\b(year\s*=\s*\d{4}|year\s+in\s*\(|month\s*=\s*\d{1,2}|month\s+in\s*\()"
        return re.search(pat, sql) is not None

    def _sql_has_year_filter(self, sql: str) -> bool:
        return re.search(r"(?is)\byear\s*(=|in\s*\()", sql) is not None

    def _sql_has_month_filter(self, sql: str) -> bool:
        return re.search(r"(?is)\bmonth\s*(=|in\s*\()", sql) is not None

    def _rewrite_growth_to_latest(self, sql: str, table_name: str) -> str:
        """Rewrite hard-coded growth month pairs to latest available periods.
        Looks for CASE WHEN month = X THEN ... constructs and IN (X,Y) lists.
        If such constructs are not found, inject WHERE year/month filters for latest two periods.
        """
        try:
            periods = self.db_manager.get_latest_periods(table_name, limit=2)
            if not periods:
                return sql
            (y1, m1) = periods[0]
            # Compute previous period
            if len(periods) >= 2:
                (y0, m0) = periods[1]
            else:
                # Derive previous month when only one period in table
                if m1 == 1:
                    y0, m0 = (y1 - 1, 12)
                else:
                    y0, m0 = (y1, m1 - 1)

            updated = sql
            # Replace month IN (...) pairs commonly appearing as (10, 11)
            updated = re.sub(r"(?is)month\s+in\s*\(\s*\d+\s*,\s*\d+\s*\)", f"month IN ({m0}, {m1})", updated)
            # Replace CASE WHEN month = X/ Y patterns
            updated = re.sub(r"(?is)CASE\s+WHEN\s+month\s*=\s*\d+\s+THEN", f"CASE WHEN month = {m1} THEN", updated, count=1)
            updated = re.sub(r"(?is)CASE\s+WHEN\s+month\s*=\s*\d+\s+THEN", f"CASE WHEN month = {m0} THEN", updated, count=1)
            # Replace explicit year constraints if present to latest year
            updated = re.sub(r"(?is)year\s*=\s*\d{4}", f"year = {y1}", updated)

            # If no time filters present after replacement, inject where with latest two
            if not self._sql_has_time_filters(updated):
                updated = self._inject_time_filter_into_sql(updated, year=y1, months=[m0, m1])

            return updated
        except Exception as e:
            logger.warning(f"_rewrite_growth_to_latest failed: {e}")
            return sql

    def _rewrite_growth_to_specific(self, sql: str, table_name: str, latest_year: int, latest_month: int) -> str:
        """Rewrite growth SQL to target a specific latest month and its previous month in the same year when available,
        otherwise roll over to prior year for previous month. Injects month IN (prev, latest) and year filter if missing.
        """
        try:
            # Compute previous month/year
            if latest_month == 1:
                prev_month = 12
                prev_year = latest_year - 1
            else:
                prev_month = latest_month - 1
                prev_year = latest_year

            updated = sql
            # Replace generic month IN pairs if present
            updated = re.sub(r"(?is)month\s+in\s*\(\s*\d+\s*,\s*\d+\s*\)", f"month IN ({prev_month}, {latest_month})", updated)
            # Replace CASE WHEN month = X THEN ... patterns
            updated = re.sub(r"(?is)CASE\s+WHEN\s+month\s*=\s*\d+\s+THEN", f"CASE WHEN month = {latest_month} THEN", updated, count=1)
            updated = re.sub(r"(?is)CASE\s+WHEN\s+month\s*=\s*\d+\s+THEN", f"CASE WHEN month = {prev_month} THEN", updated, count=1)
            # Normalize year: if any explicit year present, align latest year; otherwise inject year clause
            if self._sql_has_year_filter(updated):
                updated = re.sub(r"(?is)year\s*=\s*\d{4}", f"year = {latest_year}", updated)
            else:
                updated = self._inject_time_filter_into_sql(updated, year=latest_year, months=[prev_month, latest_month])
            return updated
        except Exception as e:
            logger.warning(f"_rewrite_growth_to_specific failed: {e}")
            return sql

    def _inject_time_filter_into_sql(self, sql: str, year: int, months: List[int]) -> str:
        """Insert year/month constraints into SQL, preserving existing clauses order."""
        try:
            clauses = []
            if year:
                clauses.append(f"year = {year}")
            if months:
                clauses.append("month IN (" + ", ".join(str(m) for m in sorted(set(months))) + ")")
            if not clauses:
                return sql
            where_clause = " AND ".join(clauses)
            # Identify insertion point: before GROUP BY/ORDER BY/LIMIT end
            tokens = [" GROUP BY ", " ORDER BY ", " LIMIT "]
            lower = sql.lower()
            insert_pos = len(sql)
            for token in tokens:
                pos = lower.find(token.lower())
                if pos != -1:
                    insert_pos = min(insert_pos, pos)
            head = sql[:insert_pos]
            tail = sql[insert_pos:]

            if re.search(r"(?is)\bwhere\b", head):
                head = re.sub(r"(?is)\bwhere\b", "WHERE", head, count=1)  # normalize case
                head = re.sub(r"(?is)WHERE", f"WHERE {where_clause} AND", head, count=1)
            else:
                # Trim trailing semicolon in head, will re-add at end of pipeline
                head = head.rstrip("; ")
                head += f" WHERE {where_clause}"
            return head + tail
        except Exception as e:
            logger.warning(f"_inject_time_filter_into_sql failed: {e}")
            return sql

    def _extract_periods_from_nlq(self, nlq_lower: str) -> Tuple[Optional[int], List[int]]:
        """Extract year and months from the user's natural language question.
        - Detect 4-digit years
        - Detect quarters Q1..Q4 and map to months
        - Detect month names/abbreviations
        Returns (year, months_list)
        """
        year: Optional[int] = None
        months: List[int] = []

        # Year
        m = re.search(r"\b(20\d{2})\b", nlq_lower)
        if m:
            try:
                year = int(m.group(1))
            except Exception:
                year = None

        # Quarter
        qm = re.search(r"\bq([1-4])\b", nlq_lower)
        if qm:
            q = int(qm.group(1))
            q_map = {1: [1,2,3], 2: [4,5,6], 3: [7,8,9], 4: [10,11,12]}
            months.extend(q_map[q])

        # Month names/abbreviations
        month_name_map = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12,
        }
        for name, num in month_name_map.items():
            if re.search(rf"\b{name}\b", nlq_lower):
                months.append(num)

        # Numeric months like "month 11" sometimes present; avoid false positives on years
        for nm in re.findall(r"\bmonth\s*(=|is|:)\s*(\d{1,2})\b", nlq_lower):
            try:
                months.append(int(nm[1]))
            except Exception:
                pass

        # Dedup and sort
        months = sorted(set([m for m in months if 1 <= m <= 12]))
        return year, months

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

    def _strip_unintended_region_filter(self, nlq: str, sql: str) -> str:
        """Remove any region filter from SQL if the original NLQ did not explicitly mention a region."""
        try:
            nlq_lower = nlq.lower()
            
            # Check if NLQ explicitly mentions a region
            region_terms = ["region", "north", "south", "east", "west", "central"]
            explicit_region = any(term in nlq_lower for term in region_terms)
            
            if not explicit_region:
                # Remove region filters from SQL (including LIKE patterns)
                # Pattern: region = 'X' or region IN ('X', 'Y') or region LIKE 'X%'
                sql = re.sub(r"(?i)\s+AND\s+region\s*=\s*['\"][^'\"]*['\"]", "", sql)
                sql = re.sub(r"(?i)\s+AND\s+region\s+IN\s*\([^)]+\)", "", sql)
                sql = re.sub(r"(?i)\s+AND\s+region\s+LIKE\s*['\"][^'\"]*['\"]", "", sql)
                sql = re.sub(r"(?i)\s+WHERE\s+region\s*=\s*['\"][^'\"]*['\"]\s+AND", " WHERE", sql)
                sql = re.sub(r"(?i)\s+WHERE\s+region\s*=\s*['\"][^'\"]*['\"]\s*$", "", sql)
                sql = re.sub(r"(?i)\s+WHERE\s+region\s+IN\s*\([^)]*\)\s+AND", " WHERE", sql)
                sql = re.sub(r"(?i)\s+WHERE\s+region\s+IN\s*\([^)]*\)\s*$", "", sql)
                sql = re.sub(r"(?i)\s+WHERE\s+region\s+LIKE\s*['\"][^'\"]*['\"]\s+AND", " WHERE", sql)
                sql = re.sub(r"(?i)\s+WHERE\s+region\s+LIKE\s*['\"][^'\"]*['\"]\s*$", "", sql)
                
                # Clean up empty WHERE clauses
                sql = re.sub(r"\s+WHERE\s+AND", " WHERE", sql)
                sql = re.sub(r"\s+WHERE\s*$", "", sql)
                
                logger.info("Removed unintended region filter from SQL")
            
            return sql
        except Exception as e:
            logger.warning(f"_strip_unintended_region_filter failed: {e}")
            return sql

    def _rewrite_window_functions_to_duckdb(self, sql: str) -> str:
        """Rewrite window function SQL to DuckDB-compatible syntax using CTEs and CASE statements."""
        try:
            sql_lower = sql.lower()
            
            # Check if SQL contains window functions
            window_patterns = [
                r"\blag\s*\(", r"\blead\s*\(", r"\brow_number\s*\(", 
                r"\brank\s*\(", r"\bdense_rank\s*\(", r"\bover\s*\("
            ]
            
            if not any(re.search(pattern, sql_lower) for pattern in window_patterns):
                return sql  # No window functions found
            
            logger.info("Detected window functions, rewriting to DuckDB-compatible syntax")
            
            # Handle LAG/LEAD functions for growth analysis
            if re.search(r"\blag\s*\(", sql_lower) or re.search(r"\blead\s*\(", sql_lower):
                return self._rewrite_lag_lead_to_case(sql)
            
            # Handle ROW_NUMBER/RANK functions
            if re.search(r"\brow_number\s*\(", sql_lower) or re.search(r"\brank\s*\(", sql_lower):
                return self._rewrite_ranking_to_duckdb(sql)
            
            # Generic window function removal
            return self._remove_window_functions(sql)
            
        except Exception as e:
            logger.warning(f"_rewrite_window_functions_to_duckdb failed: {e}")
            return sql

    def _rewrite_lag_lead_to_case(self, sql: str) -> str:
        """Rewrite LAG/LEAD window functions to CASE-based month comparisons."""
        try:
            # Extract table name and basic structure
            table_match = re.search(r"FROM\s+(\w+)", sql, re.IGNORECASE)
            if not table_match:
                return sql
            
            table_name = table_match.group(1)
            
            # Look for common growth patterns with LAG
            # Pattern: LAG(SUM(sales)) AS total_sales
            if re.search(r"lag\s*\(\s*sum\s*\(\s*sales\s*\)\s*\)", sql_lower := sql.lower()):
                # This is likely a growth query comparing months
                return self._build_growth_sql_from_lag(sql, table_name)
            
            # Pattern: LAG(SUM(sales)) OVER (ORDER BY month)
            if re.search(r"lag\s*\(\s*sum\s*\(\s*sales\s*\)\s*\)", sql_lower):
                return self._build_growth_sql_from_lag(sql, table_name)
            
            # Generic LAG/LEAD removal
            return self._remove_lag_lead_functions(sql)
            
        except Exception as e:
            logger.warning(f"_rewrite_lag_lead_to_case failed: {e}")
            return self._remove_lag_lead_functions(sql)

    def _build_growth_sql_from_lag(self, sql: str, table_name: str) -> str:
        """Build growth SQL from LAG-based window function."""
        try:
            # Extract dimension from PARTITION BY if present
            partition_match = re.search(r"partition\s+by\s+(\w+)", sql, re.IGNORECASE)
            dimension = partition_match.group(1) if partition_match else "region"
            
            # Extract time filters if present
            year_match = re.search(r"year\s*=\s*(\d{4})", str(sql))
            year = int(year_match.group(1)) if year_match else 2024
            
            month_match = re.search(r"month\s*=\s*(\d{1,2})", str(sql))
            month = int(month_match.group(1)) if month_match else 12
            
            # Build DuckDB-compatible growth SQL
            prev_month = month - 1 if month > 1 else 12
            prev_year = year if month > 1 else year - 1
            
            sql = f"""WITH monthly_sales AS (
    SELECT {dimension}, month, SUM(sales) AS total_sales
    FROM {table_name}
    WHERE year = {year} AND month IN ({prev_month}, {month})
    GROUP BY {dimension}, month
)
SELECT {dimension},
       (SUM(CASE WHEN month = {month} THEN total_sales END) -
        SUM(CASE WHEN month = {prev_month} THEN total_sales END)) * 100.0 /
        NULLIF(SUM(CASE WHEN month = {prev_month} THEN total_sales END), 0) AS sales_growth_percentage
FROM monthly_sales
GROUP BY {dimension}
ORDER BY sales_growth_percentage DESC
LIMIT 1;"""
            
            return sql
            
        except Exception as e:
            logger.warning(f"_build_growth_sql_from_lag failed: {e}")
            return self._remove_lag_lead_functions(sql)

    def _remove_lag_lead_functions(self, sql: str) -> str:
        """Remove LAG/LEAD functions and simplify the query."""
        try:
            # Remove LAG/LEAD function calls
            sql = re.sub(r"lag\s*\(\s*[^)]+\s*\)", "0", sql, flags=re.IGNORECASE)
            sql = re.sub(r"lead\s*\(\s*[^)]+\s*\)", "0", sql, flags=re.IGNORECASE)
            
            # Remove OVER clauses
            sql = re.sub(r"over\s*\(\s*[^)]*\s*\)", "", sql, flags=re.IGNORECASE)
            
            # Clean up empty parentheses
            sql = re.sub(r"\(\s*\)", "", sql)
            
            # Remove empty ORDER BY clauses
            sql = re.sub(r"order\s+by\s*$", "", sql, flags=re.IGNORECASE)
            
            return sql
            
        except Exception as e:
            logger.warning(f"_remove_lag_lead_functions failed: {e}")
            return sql

    def _remove_window_functions(self, sql: str) -> str:
        """Remove all window functions and simplify the query."""
        try:
            # Remove common window functions
            window_funcs = [
                r"row_number\s*\(\s*\)", r"rank\s*\(\s*\)", r"dense_rank\s*\(\s*\)",
                r"ntile\s*\(\s*\d+\s*\)", r"percent_rank\s*\(\s*\)", r"cume_dist\s*\(\s*\)"
            ]
            
            for pattern in window_funcs:
                sql = re.sub(pattern, "1", sql, flags=re.IGNORECASE)
            
            # Remove OVER clauses
            sql = re.sub(r"over\s*\(\s*[^)]*\s*\)", "", sql, flags=re.IGNORECASE)
            
            # Clean up empty parentheses and clauses
            sql = re.sub(r"\(\s*\)", "", sql)
            sql = re.sub(r"order\s+by\s*$", "", sql, flags=re.IGNORECASE)
            
            return sql
            
        except Exception as e:
            logger.warning(f"_remove_window_functions failed: {e}")
            return sql

    def _rewrite_ranking_to_duckdb(self, sql: str) -> str:
        """Rewrite ROW_NUMBER/RANK functions to DuckDB-compatible syntax."""
        try:
            # Extract table name and basic structure
            table_match = re.search(r"FROM\s+(\w+)", sql, re.IGNORECASE)
            if not table_match:
                return sql
            
            table_name = table_match.group(1)
            
            # Look for common ranking patterns
            if re.search(r"row_number\s*\(\s*\)", sql.lower()):
                # Convert ROW_NUMBER() OVER (ORDER BY ...) to simple ORDER BY with LIMIT
                return self._convert_row_number_to_limit(sql, table_name)
            
            if re.search(r"rank\s*\(\s*\)", sql.lower()):
                # Convert RANK() to simple ORDER BY
                return self._convert_rank_to_order(sql, table_name)
            
            # Generic ranking removal
            return self._remove_ranking_functions(sql)
            
        except Exception as e:
            logger.warning(f"_rewrite_ranking_to_duckdb failed: {e}")
            return self._remove_ranking_functions(sql)

    def _convert_row_number_to_limit(self, sql: str, table_name: str) -> str:
        """Convert ROW_NUMBER() OVER (ORDER BY ...) to simple ORDER BY with LIMIT."""
        try:
            # Extract ORDER BY clause from OVER
            order_match = re.search(r"over\s*\(\s*order\s+by\s+([^)]+)\)", sql, re.IGNORECASE)
            if order_match:
                order_clause = order_match.group(1).strip()
                
                # Remove the OVER clause and ROW_NUMBER
                sql = re.sub(r"row_number\s*\(\s*\)\s*over\s*\(\s*order\s+by\s+[^)]+\)", "", sql, flags=re.IGNORECASE)
                
                # Add ORDER BY and LIMIT
                sql = re.sub(r"(\bFROM\s+\w+.*?)(\s*;?\s*)$", r"\1 ORDER BY " + order_clause + r" LIMIT 1\2", sql, flags=re.IGNORECASE | re.DOTALL)
                
                return sql
            
            # Fallback: remove ROW_NUMBER and add simple ORDER BY
            sql = re.sub(r"row_number\s*\(\s*\)\s*over\s*\(\s*[^)]*\)", "", sql, flags=re.IGNORECASE)
            sql = re.sub(r"(\bFROM\s+\w+.*?)(\s*;?\s*)$", r"\1 ORDER BY sales DESC LIMIT 1\2", sql, flags=re.IGNORECASE | re.DOTALL)
            
            return sql
            
        except Exception as e:
            logger.warning(f"_convert_row_number_to_limit failed: {e}")
            return self._remove_ranking_functions(sql)

    def _convert_rank_to_order(self, sql: str, table_name: str) -> str:
        """Convert RANK() functions to simple ORDER BY."""
        try:
            # Extract ORDER BY clause from OVER
            order_match = re.search(r"over\s*\(\s*order\s+by\s+([^)]+)\)", sql, re.IGNORECASE)
            if order_match:
                order_clause = order_match.group(1).strip()
                
                # Remove the OVER clause and RANK
                sql = re.sub(r"rank\s*\(\s*\)\s*over\s*\(\s*order\s+by\s+[^)]+\)", "", sql, flags=re.IGNORECASE)
                
                # Add ORDER BY
                sql = re.sub(r"(\bFROM\s+\w+.*?)(\s*;?\s*)$", r"\1 ORDER BY " + order_clause + r"\2", sql, flags=re.IGNORECASE | re.DOTALL)
                
                return sql
            
            # Fallback: remove RANK and add simple ORDER BY
            sql = re.sub(r"rank\s*\(\s*\)\s*over\s*\(\s*[^)]*\)", "", sql, flags=re.IGNORECASE)
            sql = re.sub(r"(\bFROM\s+\w+.*?)(\s*;?\s*)$", r"\1 ORDER BY sales DESC\2", sql, flags=re.IGNORECASE | re.DOTALL)
            
            return sql
            
        except Exception as e:
            logger.warning(f"_convert_rank_to_order failed: {e}")
            return self._remove_ranking_functions(sql)

    def _remove_ranking_functions(self, sql: str) -> str:
        """Remove ranking functions and simplify the query."""
        try:
            # Remove ranking function calls
            sql = re.sub(r"rank\s*\(\s*\)", "1", sql, flags=re.IGNORECASE)
            sql = re.sub(r"row_number\s*\(\s*\)", "1", sql, flags=re.IGNORECASE)
            sql = re.sub(r"dense_rank\s*\(\s*\)", "1", sql, flags=re.IGNORECASE)
            
            # Remove OVER clauses
            sql = re.sub(r"over\s*\(\s*[^)]*\s*\)", "", sql, flags=re.IGNORECASE)
            
            # Clean up empty parentheses and clauses
            sql = re.sub(r"\(\s*\)", "", sql)
            sql = re.sub(r"order\s+by\s*$", "", sql, flags=re.IGNORECASE)
            
            return sql
            
        except Exception as e:
            logger.warning(f"_remove_ranking_functions failed: {e}")
            return sql

    def _build_ultimate_fallback_sql(self, nlq: str, table_name: str) -> str:
        """Ultimate fallback that generates a safe, simple SQL query when all else fails."""
        try:
            logger.info("Building ultimate fallback SQL")
            
            # Get table schema
            table_info = self.db_manager.table_schemas.get(table_name, {})
            columns = table_info.get("columns", [])
            
            if not columns:
                return None
            
            # Find available dimensions and metrics
            available_dims = []
            available_metrics = []
            
            for col in columns:
                col_lower = col.lower()
                if col_lower in ["region", "city", "area", "territory", "distributor", "route", "brand", "sku", "customer"]:
                    available_dims.append(col)
                elif col_lower in ["sales", "mro", "mto", "target", "primary sales"]:
                    available_metrics.append(col)
            
            # Default values
            dimension = available_dims[0] if available_dims else "region"
            metric = available_metrics[0] if available_metrics else "sales"
            
            # Get latest available period
            try:
                periods = self.db_manager.get_latest_periods(table_name, limit=1)
                if periods:
                    year, month = periods[0]
                    time_filter = f"WHERE year = {year} AND month = {month}"
                else:
                    time_filter = ""
            except Exception:
                time_filter = ""
            
            # Build safe SQL
            sql = f"""SELECT {self.db_manager._quote_identifier(dimension)}, 
       SUM({self.db_manager._quote_identifier(metric)}) AS total_{metric.replace(' ', '_')}
FROM {table_name}
{time_filter}
GROUP BY {self.db_manager._quote_identifier(dimension)}
ORDER BY total_{metric.replace(' ', '_')} DESC
LIMIT 5;"""
            
            logger.info(f"Generated ultimate fallback SQL: {sql}")
            return sql
            
        except Exception as e:
            logger.error(f"Ultimate fallback failed: {e}")
            return None
