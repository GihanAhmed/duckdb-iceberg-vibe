"""
Performance Benchmarking for Space Analytics Demo

This script benchmarks query performance across different storage formats
(CSV, Parquet, Iceberg) and generates comprehensive performance reports.
"""
# pylint: disable=broad-exception-caught
# pylint: disable=line-too-long
import time
import statistics
import logging
from pathlib import Path
from typing import Dict, Any
import duckdb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceBenchmarker:
    """
    Benchmarks query performance across different storage formats.

    Provides functionality for timing queries, comparing storage formats,
    analyzing storage efficiency, and generating performance reports.
    """

    def __init__(self):
        """Initialize the performance benchmarker."""
        self.results = {}
        self.connections = {}
        self.efficiency_results = {}
        self._setup_connections()

    def _setup_connections(self) -> None:
        """Setup database connections for benchmarking."""
        try:
            # Main DuckDB connection
            if Path("space_analytics.db").exists():
                self.connections['main'] = duckdb.connect("space_analytics.db")
                logger.info("✓ Connected to main DuckDB database")
            else:
                self.connections['main'] = duckdb.connect(":memory:")
                logger.warning("Main database not found, using in-memory connection")

            # Memory connection for file-based queries
            self.connections['memory'] = duckdb.connect(":memory:")

            # Install required extensions
            for conn in self.connections.values():
                for ext in ['httpfs', 'parquet', 'iceberg']:
                    try:
                        conn.execute(f"INSTALL {ext}")
                        conn.execute(f"LOAD {ext}")
                    except Exception as e:
                        logger.warning("Failed to install/load extension %s: %s", ext, e)

        except Exception as e:
            logger.error("Failed to setup connections: %s", e)
            raise

    def benchmark_query(self, query: str, connection: duckdb.DuckDBPyConnection,
                       iterations: int = 5, warmup: int = 1) -> Dict[str, float]:
        """
        Benchmark a single query multiple times.

        Args:
            query: SQL query to benchmark
            connection: DuckDB connection to use
            iterations: Number of timing iterations
            warmup: Number of warmup runs (not timed)

        Returns:
            Dictionary with timing statistics
        """
        logger.info("Benchmarking query with %d iterations...", iterations)

        try:
            # Warmup runs
            for _ in range(warmup):
                connection.execute(query).fetchall()

            # Timed runs
            times = []
            for i in range(iterations):
                start = time.perf_counter()
                result = connection.execute(query).fetchall()
                end = time.perf_counter()

                execution_time = end - start
                times.append(execution_time)

                logger.debug("  Iteration %d: %.4fs", i+1, execution_time)

            # Calculate statistics
            stats = {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0.0,
                "iterations": iterations,
                "result_rows": len(result) if result else 0
            }

            logger.info("  Mean: %.4fs, Median: %.4fs", stats['mean'], stats['median'])
            return stats

        except Exception as e:
            logger.error("Query benchmark failed: %s", e)
            return {"error": str(e)}

    def compare_storage_formats(self) -> Dict[str, Dict[str, Any]]:
        """Compare query performance across storage formats."""
        logger.info("Comparing storage format performance...")

        format_results = {}

        # Define test queries for each format
        format_configs = self._get_format_configurations()

        for format_name, config in format_configs.items():
            logger.info("\nBenchmarking %s format...", format_name)
            format_results[format_name] = {}

            if not config.get('available', False):
                logger.warning("  %s format not available, skipping", format_name)
                format_results[format_name]['status'] = 'unavailable'
                continue

            try:
                connection = config['connection']
                setup_queries = config.get('setup_queries', [])

                # Execute setup queries
                for setup_query in setup_queries:
                    connection.execute(setup_query)

                # Benchmark each test query
                for query_name, query_template in BENCHMARK_QUERIES.items():
                    logger.info("  Testing %s...", query_name)

                    # Substitute format-specific table reference
                    query = query_template.format(table_ref=config['table_ref'])

                    try:
                        benchmark_result = self.benchmark_query(query, connection, iterations=3)
                        format_results[format_name][query_name] = benchmark_result

                        if 'error' not in benchmark_result:
                            logger.info("    ✓ %s: %.4fs", query_name, benchmark_result['mean'])
                        else:
                            logger.warning("    ✗ %s: %s", query_name, benchmark_result['error'])

                    except Exception as e:
                        logger.error("    ✗ %s: %s", query_name, e)
                        format_results[format_name][query_name] = {"error": str(e)}

                format_results[format_name]['status'] = 'completed'

            except Exception as e:
                logger.error("  Format %s benchmarking failed: %s", format_name, e)
                format_results[format_name]['status'] = 'failed'
                format_results[format_name]['error'] = str(e)

        self.results['format_comparison'] = format_results
        return format_results

    def _get_format_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration for each storage format."""
        configs = {}

        # CSV format
        csv_files = list(Path("data/raw").glob("neo_data_*.csv"))
        if csv_files:
            csv_file = str(csv_files[0])
            configs['csv'] = {
                'available': True,
                'connection': self.connections['memory'],
                'table_ref': f"read_csv_auto('{csv_file}')",
                'file_path': csv_file
            }
        else:
            configs['csv'] = {'available': False}

        # Parquet format
        parquet_files = list(Path("data/raw").glob("neo_data_*.parquet"))
        if parquet_files:
            parquet_file = str(parquet_files[0])
            configs['parquet'] = {
                'available': True,
                'connection': self.connections['memory'],
                'table_ref': f"'{parquet_file}'",
                'file_path': parquet_file
            }
        else:
            configs['parquet'] = {'available': False}

        # DuckDB table format
        if 'neo_approaches' in [r[0] for r in self.connections['main'].execute("SHOW TABLES").fetchall()]:
            configs['duckdb_table'] = {
                'available': True,
                'connection': self.connections['main'],
                'table_ref': 'neo_approaches'
            }
        else:
            configs['duckdb_table'] = {'available': False}

        # Iceberg format (true Iceberg tables)
        iceberg_data_path = Path("data/iceberg_warehouse/demo.db/neo_approaches_iceberg/data")
        if iceberg_data_path.exists():
            # Find the parquet files in Iceberg data directory
            parquet_files = list(iceberg_data_path.glob("*.parquet"))
            if parquet_files:
                # Use the Iceberg data files directly
                iceberg_file = str(parquet_files[0])  # Use first file (we only have one)
                configs['iceberg'] = {
                    'available': True,
                    'connection': self.connections['memory'],
                    'table_ref': f"'{iceberg_file}'",
                    'file_path': iceberg_file
                }
            else:
                configs['iceberg'] = {'available': False}
        else:
            configs['iceberg'] = {'available': False}

        return configs

    def storage_efficiency_analysis(self) -> Dict[str, Any]:
        """Analyze storage efficiency of different formats."""
        logger.info("Analyzing storage efficiency...")

        efficiency_results = {
            'file_sizes': {},
            'compression_ratios': {},
            'query_performance_summary': {}
        }

        try:
            # Analyze file sizes
            base_size = None
            format_configs = self._get_format_configurations()

            for format_name, config in format_configs.items():
                if not config.get('available') or 'file_path' not in config:
                    continue

                file_path = Path(config['file_path'])
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    efficiency_results['file_sizes'][format_name] = {
                        'size_mb': round(size_mb, 2),
                        'size_bytes': file_path.stat().st_size
                    }

                    # Use CSV as baseline for compression ratio
                    if format_name == 'csv':
                        base_size = size_mb
                    elif base_size:
                        compression_ratio = base_size / size_mb
                        efficiency_results['compression_ratios'][format_name] = round(compression_ratio, 2)

            # Analyze Iceberg partitioned structure
            iceberg_path = Path("data/iceberg_warehouse/neo_approaches_iceberg")
            if iceberg_path.exists():
                total_size = sum(f.stat().st_size for f in iceberg_path.rglob("*.parquet"))
                size_mb = total_size / (1024 * 1024)
                efficiency_results['file_sizes']['iceberg'] = {
                    'size_mb': round(size_mb, 2),
                    'size_bytes': total_size,
                    'partition_count': len(list(iceberg_path.glob("approach_year=*")))
                }

                if base_size:
                    compression_ratio = base_size / size_mb
                    efficiency_results['compression_ratios']['iceberg'] = round(compression_ratio, 2)

            # Summarize query performance from format comparison
            if 'format_comparison' in self.results:
                for format_name, format_results in self.results['format_comparison'].items():
                    if format_results.get('status') == 'completed':
                        avg_times = []
                        for query_name, query_result in format_results.items():
                            if query_name != 'status' and isinstance(query_result, dict) and 'mean' in query_result:
                                avg_times.append(query_result['mean'])

                        if avg_times:
                            efficiency_results['query_performance_summary'][format_name] = {
                                'avg_query_time': round(statistics.mean(avg_times), 4),
                                'total_queries': len(avg_times)
                            }

            logger.info("✓ Storage efficiency analysis completed")
            return efficiency_results

        except Exception as e:
            logger.error("Storage efficiency analysis failed: %s", e)
            return {"error": str(e)}

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        logger.info("Generating performance report...")

        try:
            report_lines = [
                "# Space Analytics Performance Benchmark Report",
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Executive Summary",
                ""
            ]

            # Add format comparison summary
            if 'format_comparison' in self.results:
                report_lines.extend([
                    "### Storage Format Performance Comparison",
                    ""
                ])

                # Create performance table
                format_data = []
                for format_name, results in self.results['format_comparison'].items():
                    if results.get('status') == 'completed':
                        avg_times = []
                        for query_name, query_result in results.items():
                            if query_name != 'status' and isinstance(query_result, dict) and 'mean' in query_result:
                                avg_times.append(query_result['mean'])

                        if avg_times:
                            format_data.append({
                                'format': format_name,
                                'avg_time': statistics.mean(avg_times),
                                'queries_tested': len(avg_times)
                            })

                # Sort by performance
                format_data.sort(key=lambda x: x['avg_time'])

                report_lines.append("| Format | Avg Query Time (s) | Queries Tested | Performance Rank |")
                report_lines.append("|--------|-------------------|----------------|------------------|")

                for i, data in enumerate(format_data):
                    rank = i + 1
                    report_lines.append(f"| {data['format']} | {data['avg_time']:.4f} | {data['queries_tested']} | #{rank} |")

                report_lines.append("")

            # Add detailed results
            if 'format_comparison' in self.results:
                report_lines.extend([
                    "## Detailed Results",
                    ""
                ])

                for format_name, results in self.results['format_comparison'].items():
                    report_lines.extend([
                        f"### {format_name.upper()} Format",
                        ""
                    ])

                    if results.get('status') == 'completed':
                        for query_name, query_result in results.items():
                            if query_name != 'status' and isinstance(query_result, dict):
                                if 'mean' in query_result:
                                    report_lines.extend([
                                        f"**{query_name.replace('_', ' ').title()}:**",
                                        f"- Mean: {query_result['mean']:.4f}s",
                                        f"- Median: {query_result['median']:.4f}s",
                                        f"- Min: {query_result['min']:.4f}s",
                                        f"- Max: {query_result['max']:.4f}s",
                                        f"- Rows: {query_result.get('result_rows', 'N/A')}",
                                        ""
                                    ])
                                elif 'error' in query_result:
                                    report_lines.extend([
                                        f"**{query_name.replace('_', ' ').title()}:** ERROR",
                                        f"- {query_result['error']}",
                                        ""
                                    ])
                    else:
                        status = results.get('status', 'unknown')
                        report_lines.append(f"Status: {status}")
                        if 'error' in results:
                            report_lines.append(f"Error: {results['error']}")
                        report_lines.append("")

            # Add storage efficiency analysis
            if hasattr(self, 'efficiency_results'):
                report_lines.extend([
                    "## Storage Efficiency Analysis",
                    ""
                ])

                if 'file_sizes' in self.efficiency_results:
                    report_lines.extend([
                        "### File Sizes",
                        ""
                    ])

                    for format_name, size_info in self.efficiency_results['file_sizes'].items():
                        report_lines.append(f"- **{format_name}**: {size_info['size_mb']} MB")

                    report_lines.append("")

                if 'compression_ratios' in self.efficiency_results:
                    report_lines.extend([
                        "### Compression Ratios (vs CSV)",
                        ""
                    ])

                    for format_name, ratio in self.efficiency_results['compression_ratios'].items():
                        report_lines.append(f"- **{format_name}**: {ratio}x compression")

                    report_lines.append("")

            # Save report
            report_content = "\n".join(report_lines)
            report_file = Path("performance_report.md")

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)

            logger.info("✓ Performance report saved to %s", report_file)
            return str(report_file)

        except Exception as e:
            logger.error("Failed to generate performance report: %s", e)
            return ""

    def close(self) -> None:
        """Close all database connections."""
        for name, conn in self.connections.items():
            try:
                conn.close()
                logger.debug("✓ Closed %s connection", name)
            except Exception as e:
                logger.warning("Failed to close %s connection: %s", name, e)

        logger.info("✓ All benchmark connections closed")


# Required benchmark queries
BENCHMARK_QUERIES = {
    "simple_count": "SELECT COUNT(*) FROM {table_ref}",

    "filtered_count": "SELECT COUNT(*) FROM {table_ref} WHERE h < 20",

    "aggregation": """
        SELECT
            approach_year,
            COUNT(*) as count,
            ROUND(AVG(dist), 6) as avg_dist
        FROM {table_ref}
        WHERE approach_year IS NOT NULL
        GROUP BY approach_year
        ORDER BY approach_year DESC
        LIMIT 10
    """,

    "complex_filtering": """
        SELECT
            des,
            fullname,
            h,
            dist,
            v_rel,
            CASE
                WHEN h < 18 AND dist < 0.05 THEN 'HIGH_RISK'
                WHEN h < 22 AND dist < 0.1 THEN 'MEDIUM_RISK'
                ELSE 'LOW_RISK'
            END as risk_category
        FROM {table_ref}
        WHERE h IS NOT NULL AND dist IS NOT NULL
        ORDER BY dist ASC
        LIMIT 50
    """,

    "analytical_window": """
        SELECT
            des,
            h,
            dist,
            v_rel,
            ROW_NUMBER() OVER (ORDER BY dist ASC) as distance_rank,
            PERCENT_RANK() OVER (ORDER BY h DESC) as size_percentile
        FROM {table_ref}
        WHERE h IS NOT NULL AND dist IS NOT NULL
        ORDER BY dist ASC
        LIMIT 25
    """
}


def main():
    """Main execution function for performance benchmarking."""
    logger.info("=" * 60)
    logger.info("PERFORMANCE BENCHMARKING")
    logger.info("=" * 60)

    benchmarker = None

    try:
        # Initialize benchmarker
        benchmarker = PerformanceBenchmarker()

        # Compare storage formats
        format_results = benchmarker.compare_storage_formats()

        # Analyze storage efficiency
        efficiency_results = benchmarker.storage_efficiency_analysis()
        benchmarker.efficiency_results = efficiency_results

        # Generate performance report
        report_file = benchmarker.generate_performance_report()

        # Summary
        logger.info("\n%s\nBENCHMARKING SUMMARY\n%s", "="*60, "="*60)

        completed_formats = [name for name, results in format_results.items()
                           if results.get('status') == 'completed']

        logger.info("Formats tested: %d", len(completed_formats))
        if completed_formats:
            logger.info("  ✓ %s", ", ".join(completed_formats))

        failed_formats = [name for name, results in format_results.items()
                         if results.get('status') == 'failed']

        if failed_formats:
            logger.info("Failed formats: %s", ", ".join(failed_formats))

        if report_file:
            logger.info("Report generated: %s", report_file)

        logger.info("\n🎉 Performance benchmarking completed!")
        logger.info("Next step: python scripts/06_advanced_analytics.py")

        return True

    except Exception as e:
        logger.error("❌ Performance benchmarking failed: %s", e)
        return False
    finally:
        if benchmarker:
            benchmarker.close()


if __name__ == "__main__":
    import sys
    SUCCESS = main()
    sys.exit(0 if SUCCESS else 1)
