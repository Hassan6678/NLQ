"""
Performance benchmarking and profiling tools for the NLQ system.
"""

import time
import json
import os
import sys
import statistics
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, asdict
import pandas as pd
import psutil
import memory_profiler

from main import initialize_system, NLQSystem
from config import config


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    query: str
    execution_time: float
    memory_usage_mb: float
    result_rows: int
    cache_hit: bool
    sql_generated: str
    success: bool
    error_message: str = ""


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, system: NLQSystem):
        self.system = system
        self.results: List[BenchmarkResult] = []
        self.start_memory = 0
        self.peak_memory = 0
    
    def run_query_benchmark(self, queries: List[Dict[str, str]], iterations: int = 3) -> Dict[str, Any]:
        """Run benchmark on a set of queries with multiple iterations."""
        print(f"🚀 Running benchmark with {len(queries)} queries, {iterations} iterations each")
        print("=" * 70)
        
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        for query_info in queries:
            query_name = query_info["name"]
            query_text = query_info["query"]
            
            print(f"\n📊 Testing: {query_name}")
            print(f"Query: {query_text}")
            
            query_results = []
            
            for iteration in range(iterations):
                print(f"  Iteration {iteration + 1}/{iterations}...", end=" ")
                
                try:
                    start_time = time.time()
                    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    result = self.system.query(query_text)
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    benchmark_result = BenchmarkResult(
                        test_name=f"{query_name}_iter_{iteration + 1}",
                        query=query_text,
                        execution_time=execution_time,
                        memory_usage_mb=memory_delta,
                        result_rows=result.row_count,
                        cache_hit=result.from_cache,
                        sql_generated=result.sql_query,
                        success=True
                    )
                    
                    query_results.append(benchmark_result)
                    self.results.append(benchmark_result)
                    
                    print(f"✅ {execution_time:.3f}s ({result.row_count:,} rows)")
                    
                    # Update peak memory
                    self.peak_memory = max(self.peak_memory, end_memory)
                    
                except Exception as e:
                    error_result = BenchmarkResult(
                        test_name=f"{query_name}_iter_{iteration + 1}",
                        query=query_text,
                        execution_time=0,
                        memory_usage_mb=0,
                        result_rows=0,
                        cache_hit=False,
                        sql_generated="",
                        success=False,
                        error_message=str(e)
                    )
                    
                    query_results.append(error_result)
                    self.results.append(error_result)
                    
                    print(f"❌ Error: {e}")
            
            # Calculate statistics for this query
            successful_results = [r for r in query_results if r.success]
            if successful_results:
                times = [r.execution_time for r in successful_results]
                memory_usage = [r.memory_usage_mb for r in successful_results]
                
                print(f"  📈 Statistics:")
                print(f"    Avg time: {statistics.mean(times):.3f}s")
                print(f"    Min time: {min(times):.3f}s")
                print(f"    Max time: {max(times):.3f}s")
                if len(times) > 1:
                    print(f"    Std dev: {statistics.stdev(times):.3f}s")
                print(f"    Avg memory: {statistics.mean(memory_usage):.1f}MB")
                print(f"    Cache hits: {sum(1 for r in successful_results if r.cache_hit)}/{len(successful_results)}")
        
        return self.generate_report()
    
    def run_scalability_test(self, base_query: str, data_sizes: List[int]) -> Dict[str, Any]:
        """Test performance scaling with different data sizes."""
        print(f"🔬 Running scalability test with data sizes: {data_sizes}")
        print("=" * 70)
        
        scalability_results = []
        
        for size in data_sizes:
            print(f"\n📊 Testing with {size:,} rows")
            
            # Modify query to limit results for testing
            test_query = f"{base_query} LIMIT {size}"
            
            try:
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                result = self.system.query(test_query)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                scalability_results.append({
                    "data_size": size,
                    "execution_time": execution_time,
                    "memory_usage_mb": memory_delta,
                    "rows_per_second": result.row_count / execution_time if execution_time > 0 else 0,
                    "success": True
                })
                
                print(f"  ✅ {execution_time:.3f}s, {result.row_count:,} rows, {memory_delta:.1f}MB")
                
            except Exception as e:
                scalability_results.append({
                    "data_size": size,
                    "execution_time": 0,
                    "memory_usage_mb": 0,
                    "rows_per_second": 0,
                    "success": False,
                    "error": str(e)
                })
                print(f"  ❌ Error: {e}")
        
        return {
            "scalability_results": scalability_results,
            "summary": self._analyze_scalability(scalability_results)
        }
    
    def run_memory_profile(self, query: str, profile_duration: int = 60) -> Dict[str, Any]:
        """Profile memory usage during query execution."""
        print(f"🧠 Running memory profile for {profile_duration}s")
        print("=" * 70)
        
        @memory_profiler.profile
        def profiled_query():
            return self.system.query(query)
        
        try:
            start_time = time.time()
            result = profiled_query()
            end_time = time.time()
            
            return {
                "query": query,
                "execution_time": end_time - start_time,
                "result_rows": result.row_count,
                "memory_profile_available": True,
                "note": "Check console output for detailed memory profile"
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "memory_profile_available": False
            }
    
    def _analyze_scalability(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scalability test results."""
        successful_results = [r for r in results if r["success"]]
        
        if not successful_results:
            return {"error": "No successful results to analyze"}
        
        data_sizes = [r["data_size"] for r in successful_results]
        execution_times = [r["execution_time"] for r in successful_results]
        memory_usage = [r["memory_usage_mb"] for r in successful_results]
        
        # Calculate scaling factors
        time_scaling = execution_times[-1] / execution_times[0] if len(execution_times) > 1 else 1
        memory_scaling = memory_usage[-1] / memory_usage[0] if len(memory_usage) > 1 and memory_usage[0] > 0 else 1
        data_scaling = data_sizes[-1] / data_sizes[0] if len(data_sizes) > 1 else 1
        
        return {
            "data_size_range": f"{min(data_sizes):,} - {max(data_sizes):,} rows",
            "time_scaling_factor": time_scaling,
            "memory_scaling_factor": memory_scaling,
            "data_scaling_factor": data_scaling,
            "efficiency_ratio": data_scaling / time_scaling if time_scaling > 0 else 0,
            "linear_scaling": abs(time_scaling - data_scaling) < 0.5,
            "average_rows_per_second": statistics.mean([r["rows_per_second"] for r in successful_results])
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        if not successful_results:
            return {"error": "No successful benchmark results"}
        
        execution_times = [r.execution_time for r in successful_results]
        memory_usage = [r.memory_usage_mb for r in successful_results]
        result_rows = [r.result_rows for r in successful_results]
        cache_hits = sum(1 for r in successful_results if r.cache_hit)
        
        report = {
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": len(successful_results),
                "failed_tests": len(failed_results),
                "success_rate": len(successful_results) / len(self.results) * 100,
                "cache_hit_rate": cache_hits / len(successful_results) * 100 if successful_results else 0
            },
            "performance_metrics": {
                "execution_time": {
                    "mean": statistics.mean(execution_times),
                    "median": statistics.median(execution_times),
                    "min": min(execution_times),
                    "max": max(execution_times),
                    "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                },
                "memory_usage": {
                    "mean": statistics.mean(memory_usage),
                    "median": statistics.median(memory_usage),
                    "min": min(memory_usage),
                    "max": max(memory_usage),
                    "peak_total": self.peak_memory,
                    "baseline": self.start_memory
                },
                "throughput": {
                    "total_rows_processed": sum(result_rows),
                    "average_rows_per_query": statistics.mean(result_rows),
                    "rows_per_second": sum(result_rows) / sum(execution_times) if sum(execution_times) > 0 else 0
                }
            },
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                "memory_available_gb": psutil.virtual_memory().available / 1024**3,
                "config": {
                    "chunk_size": config.database.chunk_size,
                    "connection_pool_size": config.database.connection_pool_size,
                    "model_context": config.model.n_ctx,
                    "model_threads": config.model.n_threads
                }
            },
            "detailed_results": [asdict(r) for r in self.results]
        }
        
        if failed_results:
            report["failures"] = [
                {"test": r.test_name, "query": r.query, "error": r.error_message}
                for r in failed_results
            ]
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save benchmark report to file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}.json"
        
        os.makedirs("benchmarks", exist_ok=True)
        filepath = os.path.join("benchmarks", filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Benchmark report saved to: {filepath}")
        return filepath


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark suite."""
    print("🎯 Starting Comprehensive NLQ System Benchmark")
    print("=" * 70)
    
    # Initialize system
    try:
        system = initialize_system()
        
        # Load test data
        if os.path.exists("sales.csv"):
            print("📁 Loading test data...")
            load_result = system.load_data("sales.csv", "sales_data")
            print(f"✅ Loaded {load_result['total_rows']:,} rows")
        else:
            print("❌ sales.csv not found. Creating sample data...")
            create_sample_data()
            load_result = system.load_data("sample_sales.csv", "sales_data")
            print(f"✅ Loaded {load_result['total_rows']:,} rows from sample data")
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    benchmark = PerformanceBenchmark(system)
    
    # Define test queries
    test_queries = [
        {
            "name": "Simple Aggregation",
            "query": "What are the total sales?"
        },
        {
            "name": "Grouped Aggregation",
            "query": "Show me total sales by region"
        },
        {
            "name": "Filtered Aggregation",
            "query": "What were the total sales in Q3?"
        },
        {
            "name": "Top N Query",
            "query": "Show me the top 5 regions by sales"
        },
        {
            "name": "Complex Analysis",
            "query": "What is the average sales per quarter for each region?"
        }
    ]
    
    # Run main benchmark
    print("\n🏃‍♂️ Running Query Benchmark...")
    main_report = benchmark.run_query_benchmark(test_queries, iterations=3)
    
    # Run scalability test
    print("\n📈 Running Scalability Test...")
    scalability_report = benchmark.run_scalability_test(
        "SELECT * FROM sales_data",
        [1000, 5000, 10000, 50000]
    )
    
    # Combine reports
    final_report = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "main_benchmark": main_report,
        "scalability_test": scalability_report,
        "system_performance": system.get_performance_report()
    }
    
    # Save report
    report_file = benchmark.save_report(final_report)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 70)
    
    if "summary" in main_report:
        summary = main_report["summary"]
        print(f"✅ Success Rate: {summary['success_rate']:.1f}%")
        print(f"🎯 Cache Hit Rate: {summary['cache_hit_rate']:.1f}%")
        
        if "performance_metrics" in main_report:
            perf = main_report["performance_metrics"]
            print(f"⚡ Avg Execution Time: {perf['execution_time']['mean']:.3f}s")
            print(f"🧠 Avg Memory Usage: {perf['memory_usage']['mean']:.1f}MB")
            print(f"📈 Throughput: {perf['throughput']['rows_per_second']:.0f} rows/sec")
    
    print(f"📄 Full report: {report_file}")
    
    return final_report


def create_sample_data():
    """Create sample sales data for testing."""
    import random
    
    regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    
    data = []
    for _ in range(100000):  # 100K rows for testing
        data.append({
            "region": random.choice(regions),
            "quarter": random.choice(quarters),
            "sales": random.randint(1000, 50000)
        })
    
    df = pd.DataFrame(data)
    df.to_csv("sample_sales.csv", index=False)
    print("✅ Created sample_sales.csv with 100K rows")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        create_sample_data()
    else:
        run_comprehensive_benchmark()