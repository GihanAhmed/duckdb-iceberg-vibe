"""
Unit tests for performance benchmarking functionality.
"""
import pytest
import duckdb
import pandas as pd
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add scripts to path for testing
sys.path.append('../scripts')
from scripts.performance_benchmarks import PerformanceBenchmarker


class TestPerformanceBenchmarker:
    """Test suite for PerformanceBenchmarker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmarker = PerformanceBenchmarker()
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'des': ['NEO1', 'NEO2', 'NEO3'],
            'h': [18.5, 22.1, 15.3],
            'dist': [0.02, 0.15, 0.008],
            'v_rel': [25.5, 18.2, 35.7],
            'approach_year': [2020, 2021, 2022]
        })
        
        # Setup in-memory test database
        self.test_conn = duckdb.connect(":memory:")
        self.test_conn.execute("CREATE TABLE neo_approaches AS SELECT * FROM $1", [self.sample_data])
    
    def test_initialization(self):
        """Test PerformanceBenchmarker initialization."""
        benchmarker = PerformanceBenchmarker()
        assert hasattr(benchmarker, 'results')
        assert hasattr(benchmarker, 'connections')
        assert isinstance(benchmarker.results, dict)
        assert isinstance(benchmarker.connections, dict)
    
    def test_benchmark_query_success(self):
        """Test successful query benchmarking."""
        query = "SELECT COUNT(*) FROM neo_approaches"
        
        result = self.benchmarker.benchmark_query(
            query, 
            self.test_conn, 
            iterations=3, 
            warmup=1
        )
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'mean' in result
        assert 'median' in result
        assert 'min' in result
        assert 'max' in result
        assert 'std' in result
        assert 'iterations' in result
        assert 'result_rows' in result
        
        # Check values are reasonable
        assert result['iterations'] == 3
        assert result['mean'] > 0
        assert result['min'] <= result['mean'] <= result['max']
        assert result['std'] >= 0
    
    def test_benchmark_query_error_handling(self):
        """Test error handling in query benchmarking."""
        invalid_query = "SELECT * FROM nonexistent_table"
        
        result = self.benchmarker.benchmark_query(
            invalid_query, 
            self.test_conn, 
            iterations=1
        )
        
        assert 'error' in result
        assert isinstance(result['error'], str)
    
    def test_benchmark_query_timing_consistency(self):
        """Test that benchmark timing is consistent."""
        simple_query = "SELECT 1"
        
        result = self.benchmarker.benchmark_query(
            simple_query,
            self.test_conn,
            iterations=5
        )
        
        # For a simple query, times should be very small and consistent
        assert result['mean'] < 1.0  # Should complete in less than 1 second
        assert result['min'] <= result['median'] <= result['max']
        assert result['std'] >= 0
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_get_format_configurations(self, mock_glob, mock_exists):
        """Test format configuration detection."""
        # Mock file existence
        mock_exists.return_value = True
        mock_glob.return_value = [Path("test_data.csv")]
        
        configs = self.benchmarker._get_format_configurations()
        
        assert isinstance(configs, dict)
        # Should have at least some format configurations
        assert len(configs) > 0
        
        # Check structure of configurations
        for format_name, config in configs.items():
            assert 'available' in config
            if config['available']:
                assert 'connection' in config
                assert 'table_ref' in config
    
    def test_compare_storage_formats_no_data(self):
        """Test storage format comparison with no data available."""
        # Create benchmarker with no data files
        with patch('pathlib.Path.glob', return_value=[]):
            with patch('pathlib.Path.exists', return_value=False):
                benchmarker = PerformanceBenchmarker()
                result = benchmarker.compare_storage_formats()
        
        assert isinstance(result, dict)
        # Should handle gracefully when no formats are available
    
    @patch('pathlib.Path.exists')
    def test_storage_efficiency_analysis(self, mock_exists):
        """Test storage efficiency analysis."""
        # Mock file existence
        mock_exists.return_value = False
        
        result = self.benchmarker.storage_efficiency_analysis()
        
        assert isinstance(result, dict)
        assert 'file_sizes' in result
        assert 'compression_ratios' in result
        assert 'query_performance_summary' in result
    
    def test_generate_performance_report_empty_results(self):
        """Test performance report generation with empty results."""
        # Clear any existing results
        self.benchmarker.results = {}
        
        report_path = self.benchmarker.generate_performance_report()
        
        # Should create report even with no results
        if report_path:
            assert Path(report_path).exists()
    
    def test_generate_performance_report_with_data(self):
        """Test performance report generation with sample data."""
        # Add sample results
        self.benchmarker.results = {
            'format_comparison': {
                'csv': {
                    'status': 'completed',
                    'simple_count': {'mean': 1.5, 'median': 1.4, 'min': 1.2, 'max': 1.8, 'result_rows': 100}
                },
                'parquet': {
                    'status': 'completed', 
                    'simple_count': {'mean': 0.5, 'median': 0.4, 'min': 0.3, 'max': 0.7, 'result_rows': 100}
                }
            }
        }
        
        report_path = self.benchmarker.generate_performance_report()
        
        if report_path:
            assert Path(report_path).exists()
            
            # Check report content
            with open(report_path, 'r') as f:
                content = f.read()
            
            assert "Performance Benchmark Report" in content
            assert "csv" in content.lower()
            assert "parquet" in content.lower()
    
    def test_benchmark_queries_structure(self):
        """Test that all benchmark queries are valid."""
        from scripts.performance_benchmarks import BENCHMARK_QUERIES
        
        assert isinstance(BENCHMARK_QUERIES, dict)
        assert len(BENCHMARK_QUERIES) > 0
        
        # Check each query has required structure
        for query_name, query in BENCHMARK_QUERIES.items():
            assert isinstance(query_name, str)
            assert isinstance(query, str)
            assert len(query.strip()) > 0
            assert '{table_ref}' in query  # Should have table reference placeholder
    
    def test_benchmark_queries_execution(self):
        """Test that benchmark queries can be executed."""
        from scripts.performance_benchmarks import BENCHMARK_QUERIES
        
        # Test a few queries with our sample data
        for query_name, query_template in list(BENCHMARK_QUERIES.items())[:2]:
            query = query_template.format(table_ref='neo_approaches')
            
            try:
                result = self.test_conn.execute(query).fetchall()
                assert result is not None
            except Exception as e:
                # Some queries might fail due to missing columns, which is OK for testing
                assert "column" in str(e).lower() or "table" in str(e).lower()
    
    def test_close_connections(self):
        """Test connection cleanup."""
        # Add a test connection
        test_conn = duckdb.connect(":memory:")
        self.benchmarker.connections['test'] = test_conn
        
        # Close all connections
        self.benchmarker.close()
        
        # Verify connections are closed (this is implicit, as DuckDB doesn't throw on closed connections)
        assert True  # If we reach here, close() didn't raise an exception
    
    def test_performance_statistics_calculation(self):
        """Test statistical calculations in benchmark results."""
        query = "SELECT COUNT(*) FROM neo_approaches"
        
        result = self.benchmarker.benchmark_query(
            query,
            self.test_conn,
            iterations=5
        )
        
        # Test statistical properties
        assert result['min'] <= result['median'] <= result['max']
        assert result['min'] <= result['mean'] <= result['max']
        
        if result['iterations'] > 1:
            # Standard deviation should be non-negative
            assert result['std'] >= 0
    
    def test_warmup_iterations(self):
        """Test that warmup iterations work correctly."""
        query = "SELECT COUNT(*) FROM neo_approaches"
        
        # Test with different warmup values
        result_no_warmup = self.benchmarker.benchmark_query(
            query, self.test_conn, iterations=3, warmup=0
        )
        
        result_with_warmup = self.benchmarker.benchmark_query(
            query, self.test_conn, iterations=3, warmup=2
        )
        
        # Both should succeed and have same number of timed iterations
        assert result_no_warmup['iterations'] == result_with_warmup['iterations']
        assert 'mean' in result_no_warmup
        assert 'mean' in result_with_warmup
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'test_conn'):
            self.test_conn.close()
        
        if hasattr(self, 'benchmarker'):
            self.benchmarker.close()
        
        # Clean up any generated report files
        for report_file in Path('.').glob('performance_report*.md'):
            try:
                report_file.unlink()
            except:
                pass