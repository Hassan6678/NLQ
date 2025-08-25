import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd

from config import config
from helper.nlq_system import NLQSystem


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    os.makedirs(os.path.dirname(config.logging.file_path), exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config.logging.level, logging.INFO),
        format=config.logging.format,
        handlers=[
            logging.FileHandler(config.logging.file_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("nlq.main")


logger = _setup_logging()

# ----------------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------------
nlq_system: Optional[NLQSystem] = None  # Set after initialization

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for col in formatted.columns:
        series = formatted[col]
        try:
            if pd.api.types.is_integer_dtype(series):
                formatted[col] = series.map(lambda x: f"{int(x):,}" if pd.notnull(x) else "")
            elif pd.api.types.is_float_dtype(series):
                formatted[col] = series.map(lambda x: f"{float(x):,.2f}" if pd.notnull(x) else "")
        except Exception:
            continue
    return formatted


def initialize_system(skip_validation: bool = False) -> NLQSystem:
    global nlq_system
    if nlq_system:
        return nlq_system
    if not skip_validation:
        if not config.validate():
            raise RuntimeError("Configuration validation failed")
    logger.info("Initializing NLQ System…")
    logger.info(f"Configuration memory: {config.get_memory_info()}")
    nlq_system = NLQSystem()
    return nlq_system


def _load_dataset(dataset_path: str, table_name: str = "sales_data") -> bool:
    system = initialize_system()
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return False
    try:
        logger.info(f"Loading dataset: {dataset_path}")
        res = system.load_data(dataset_path, table_name)
        print(f"✅ Loaded {res['total_rows']:,} rows into '{table_name}' in {res['duration']:.2f}s")
        return True
    except Exception as e:
        logger.exception(f"Failed loading dataset: {e}")
        return False


def run_query(nlq: str, table_name: str = "sales_data") -> None:
    if not nlq_system:
        raise RuntimeError("System not initialized. Call initialize_system().")
    try:
        print(f"\n🔍 Question: {nlq}")
        print("=" * 80)
        result = nlq_system.query(nlq, table_name)
        print("\n📜 SQL:\n" + result.sql_query)
        print(f"\n📊 Rows: {result.row_count:,}")
        if result.row_count:
            display_df = result.data if result.row_count <= 20 else result.data.head(10)
            display_df = _format_dataframe_for_display(display_df)
            if result.row_count > 20:
                print("First 10 rows:")
            try:
                print(display_df.to_string(index=False))
            except Exception:
                print(display_df)
            if result.row_count > 20:
                print(f"… and {result.row_count - 10:,} more rows")
        else:
            print("(No rows returned)")
        if getattr(result, "summary_text", ""):
            print("\n📝 Summary:")
            print(result.summary_text)
        print("\n⚡ Performance:")
        print(f"  Execution: {result.execution_time:.3f}s | Memory: {result.memory_usage_mb:.1f}MB | Cache: {'Yes' if result.from_cache else 'No'}")
    except KeyboardInterrupt:
        print("\n⏹️  Query interrupted by user")
        if nlq_system.cancel_running_query():
            print("Cancelled running SQL")
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        print(f"❌ Error: {e}")


def interactive_loop(table_name: str):
    print("\n💬 Interactive NLQ mode. Type SQL? / quit / exit to leave.")
    while True:
        try:
            user_in = input("NLQ> ").strip()
            if not user_in:
                continue
            if user_in.lower() in {"quit", "exit", ":q"}:
                break
            run_query(user_in, table_name)
        except KeyboardInterrupt:
            print("\n(Use 'exit' to quit)")
            continue
        except EOFError:
            break


def example_queries() -> List[str]:
    return [
        "What were the total sales in November 2024 in Central-A?",
        "Show the top 10 cities by total sales in 2024",
        "List the top 5 brands by sales in Lahore",
        "What are the top 5 customers by sales in 2024?",
        "How many rows had stockouts in 2024?",
        "What is the average mro for productive rows in 2024?",
        "What percent of rows were assortment = TRUE in Central-A?",
        "Total sales for brand CHOCO LAVA in Lahore-A Territory in November 2024",
        "Show the top 5 routes by sales for distributor D0715 in 2024",
    ]


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Natural Language to SQL CLI")
    p.add_argument("--data", "-d", default="data/llm_dataset_v11.gz", help="Path to dataset (.csv/.gz)")
    p.add_argument("--table", "-t", default="sales_data", help="Destination table name")
    p.add_argument("--query", "-q", help="Run a single NLQ and exit")
    p.add_argument("--interactive", "-i", action="store_true", help="Enter interactive shell after loading data")
    p.add_argument("--examples", action="store_true", help="Run bundled example queries")
    p.add_argument("--skip-validation", action="store_true", help="Skip model and resource validation (dev mode)")
    p.add_argument("--no-summary", action="store_true", help="Disable LLM summarization output")
    p.add_argument("--perf", action="store_true", help="Print performance report at end")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    # Allow user to skip validation (useful if models not present yet)
    try:
        initialize_system(skip_validation=args.skip_validation)
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return 1

    data_path = args.data
    # Expand ~ and resolve
    data_path = str(Path(data_path).expanduser())
    if not _load_dataset(data_path, args.table):
        print("Dataset load failed. Aborting.")
        return 2

    # Optionally disable summaries by pointing summarizer to None (lightweight toggle)
    if args.no_summary and nlq_system and getattr(nlq_system, "llm_manager", None):
        nlq_system.llm_manager.summarizer_llm = None

    if args.query:
        run_query(args.query, args.table)

    if args.examples:
        print("\n🚀 Running example queries…")
        for q in example_queries():
            run_query(q, args.table)
            time.sleep(0.5)

    if args.interactive and not args.query:
        interactive_loop(args.table)

    if args.perf and nlq_system:
        print("\n📈 Performance Report:")
        print(json.dumps(nlq_system.get_performance_report(), indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
