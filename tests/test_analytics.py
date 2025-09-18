"""
Unit tests for advanced analytics functionality.
"""
import pytest
import pandas as pd
import numpy as np
import duckdb
import json
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add scripts to path for testing
sys.path.append('../scripts')
from scripts.advanced_analytics import SimpleAnalytics


class TestSimpleAnalytics:
    """Test suite for SimpleAnalytics class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create in-memory test database with sample data
        self.test_conn = duckdb.connect(":memory:")
        
        # Sample NEO data
        self.sample_data = pd.DataFrame({
            'des': ['NEO1', 'NEO2', 'NEO3', 'NEO4', 'NEO5'],
            'fullname': ['NEO Object 1', 'NEO Object 2', 'NEO Object 3', 'NEO Object 4', 'NEO Object 5'],
            'h': [18.5, 22.1, 15.3, 25.0, 19.8],
            'dist': [0.02, 0.15, 0.008, 0.3, 0.045],
            'v_rel': [25.5, 18.2, 35.7, 12.1, 28.3],
            'cd': ['2020-01-15', '2021-03-22', '2022-05-10', '2023-07-08', '2024-02-14'],
            'approach_year': [2020, 2021, 2022, 2023, 2024]
        })
        
        # Create table in test database
        self.test_conn.execute("CREATE TABLE neo_approaches AS SELECT * FROM $1", [self.sample_data])
        
        # Initialize analytics with test connection
        self.analytics = SimpleAnalytics(self.test_conn)
    
    def test_initialization_with_connection(self):
        """Test SimpleAnalytics initialization with provided connection."""
        analytics = SimpleAnalytics(self.test_conn)
        assert analytics.conn is not None
        assert hasattr(analytics, 'analytics_results')
        assert isinstance(analytics.analytics_results, dict)
    
    def test_initialization_without_connection(self):
        """Test SimpleAnalytics initialization without connection."""
        with patch('pathlib.Path.exists', return_value=False):
            analytics = SimpleAnalytics()
            assert analytics.conn is not None
    
    def test_calculate_risk_score_valid_inputs(self):
        """Test risk score calculation with valid inputs."""
        # Test case 1: Large, close, fast object (high risk)
        risk1 = self.analytics.calculate_risk_score(h=15.0, dist=0.01, v_rel=40.0)
        assert risk1 > 50  # Should be high risk
        
        # Test case 2: Small, distant, slow object (low risk)
        risk2 = self.analytics.calculate_risk_score(h=25.0, dist=0.5, v_rel=10.0)
        assert risk2 < risk1  # Should be lower risk
        
        # Test case 3: Moderate object
        risk3 = self.analytics.calculate_risk_score(h=20.0, dist=0.1, v_rel=25.0)
        assert 0 <= risk3 <= 100  # Should be within valid range
    
    def test_calculate_risk_score_invalid_inputs(self):
        """Test risk score calculation with invalid inputs."""
        # Test with NaN values
        risk_nan = self.analytics.calculate_risk_score(h=np.nan, dist=0.01, v_rel=25.0)
        assert risk_nan == 0.0
        
        # Test with None values
        risk_none = self.analytics.calculate_risk_score(h=None, dist=0.01, v_rel=25.0)
        assert risk_none == 0.0
        
        # Test with all invalid
        risk_all_invalid = self.analytics.calculate_risk_score(h=np.nan, dist=np.nan, v_rel=np.nan)
        assert risk_all_invalid == 0.0
    
    def test_calculate_risk_score_edge_cases(self):
        """Test risk score calculation edge cases."""
        # Very large object (h=0)
        risk_huge = self.analytics.calculate_risk_score(h=0.0, dist=0.01, v_rel=25.0)
        assert risk_huge > 50
        
        # Zero distance (theoretical minimum)
        risk_zero_dist = self.analytics.calculate_risk_score(h=20.0, dist=0.0, v_rel=25.0)
        assert risk_zero_dist > 0
        
        # Very high velocity
        risk_fast = self.analytics.calculate_risk_score(h=20.0, dist=0.1, v_rel=100.0)
        assert risk_fast <= 100  # Should be capped at 100
    
    def test_find_risky_objects(self):
        """Test finding risky objects."""
        risky_df = self.analytics.find_risky_objects(top_n=3)
        
        assert isinstance(risky_df, pd.DataFrame)
        assert len(risky_df) <= 3  # Should return at most 3 objects
        
        if not risky_df.empty:
            # Check required columns
            required_cols = ['des', 'fullname', 'h', 'distance_au', 'velocity_kms', 'risk_score', 'risk_category']
            for col in required_cols:
                assert col in risky_df.columns
            
            # Check risk scores are sorted descending
            if len(risky_df) > 1:
                risk_scores = risky_df['risk_score'].values
                assert all(risk_scores[i] >= risk_scores[i+1] for i in range(len(risk_scores)-1))
            
            # Check risk scores are valid
            assert all(0 <= score <= 100 for score in risky_df['risk_score'])
    
    def test_analyze_trends_by_year(self):
        """Test yearly trend analysis."""
        trends_df = self.analytics.analyze_trends_by_year()
        
        assert isinstance(trends_df, pd.DataFrame)
        
        if not trends_df.empty:
            # Check required columns
            required_cols = ['year', 'total_approaches', 'unique_objects', 'avg_distance_au', 'closest_approach_au']
            for col in required_cols:
                assert col in trends_df.columns
            
            # Check data types
            assert pd.api.types.is_numeric_dtype(trends_df['year'])
            assert pd.api.types.is_numeric_dtype(trends_df['total_approaches'])
            
            # Check logical constraints
            assert all(trends_df['total_approaches'] > 0)
            assert all(trends_df['unique_objects'] > 0)
            assert all(trends_df['avg_distance_au'] > 0)
            assert all(trends_df['closest_approach_au'] > 0)
    
    def test_size_distribution_analysis(self):
        """Test size distribution analysis."""
        size_df = self.analytics.size_distribution_analysis()
        
        assert isinstance(size_df, pd.DataFrame)
        
        if not size_df.empty:
            # Check required columns
            required_cols = ['size_category', 'object_count', 'closest_ever_au', 'avg_distance_au']
            for col in required_cols:
                assert col in size_df.columns
            
            # Check size categories are valid
            valid_categories = ['Very Large (>1km)', 'Large (140m-1km)', 'Medium (30-140m)', 'Small (<30m)']
            assert all(cat in valid_categories for cat in size_df['size_category'])
            
            # Check counts are positive
            assert all(size_df['object_count'] > 0)
            assert all(size_df['avg_distance_au'] > 0)
            assert all(size_df['closest_ever_au'] > 0)
    
    def test_velocity_analysis(self):
        """Test velocity analysis."""
        velocity_df = self.analytics.velocity_analysis()
        
        assert isinstance(velocity_df, pd.DataFrame)
        
        if not velocity_df.empty:
            # Check required columns
            required_cols = ['velocity_category', 'object_count', 'min_velocity', 'max_velocity', 'avg_velocity']
            for col in required_cols:
                assert col in velocity_df.columns
            
            # Check velocity categories are valid
            valid_categories = ['Slow (<10 km/s)', 'Moderate (10-20 km/s)', 'Fast (20-30 km/s)', 'Very Fast (>30 km/s)']
            assert all(cat in valid_categories for cat in velocity_df['velocity_category'])
            
            # Check velocity constraints
            for _, row in velocity_df.iterrows():
                assert row['min_velocity'] <= row['avg_velocity'] <= row['max_velocity']
                assert row['object_count'] > 0
    
    def test_close_approach_analysis(self):
        """Test close approach analysis."""
        analysis_results = self.analytics.close_approach_analysis()
        
        assert isinstance(analysis_results, dict)
        
        if 'distance_statistics' in analysis_results:
            stats = analysis_results['distance_statistics']
            
            # Check required statistics
            required_stats = ['total_approaches', 'closest_ever', 'farthest', 'mean_distance', 'median_distance']
            for stat in required_stats:
                assert stat in stats
            
            # Check logical constraints
            assert stats['total_approaches'] > 0
            assert stats['closest_ever'] <= stats['mean_distance'] <= stats['farthest']
            assert stats['closest_ever'] <= stats['median_distance'] <= stats['farthest']
        
        # Check for other analysis components
        if 'monthly_distribution' in analysis_results:
            monthly_data = analysis_results['monthly_distribution']
            assert isinstance(monthly_data, list)
            
        if 'record_holders' in analysis_results:
            records = analysis_results['record_holders']
            assert isinstance(records, list)
    
    def test_generate_analytics_summary(self):
        """Test analytics summary generation."""
        # Run some analytics first
        self.analytics.find_risky_objects(top_n=5)
        self.analytics.analyze_trends_by_year()
        
        summary = self.analytics.generate_analytics_summary()
        
        assert isinstance(summary, dict)
        
        # Check required sections
        required_sections = ['generated', 'analytics_modules', 'key_findings', 'data_quality']
        for section in required_sections:
            assert section in summary
        
        # Check data quality section
        quality = summary['data_quality']
        assert 'total_records' in quality
        assert quality['total_records'] == len(self.sample_data)
        
        # Check analytics modules section
        modules = summary['analytics_modules']
        assert isinstance(modules, list)
        
        # Check key findings
        findings = summary['key_findings']
        assert isinstance(findings, list)
    
    def test_save_results(self):
        """Test saving analytics results."""
        # Run some analytics
        self.analytics.find_risky_objects(top_n=3)
        
        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            result_path = self.analytics.save_results(temp_file)
            
            assert result_path == temp_file
            assert Path(temp_file).exists()
            
            # Verify file content
            with open(temp_file, 'r') as f:
                saved_data = json.load(f)
            
            assert isinstance(saved_data, dict)
            assert 'generated' in saved_data
            
        finally:
            # Clean up
            Path(temp_file).unlink(missing_ok=True)
    
    def test_empty_database_handling(self):
        """Test handling of empty database."""
        # Create empty database
        empty_conn = duckdb.connect(":memory:")
        empty_conn.execute("CREATE TABLE neo_approaches (des VARCHAR, h DOUBLE, dist DOUBLE, v_rel DOUBLE)")
        
        analytics = SimpleAnalytics(empty_conn)
        
        # These should handle empty data gracefully
        risky_df = analytics.find_risky_objects()
        assert risky_df.empty
        
        trends_df = analytics.analyze_trends_by_year()
        assert trends_df.empty
        
        size_df = analytics.size_distribution_analysis()
        assert size_df.empty
        
        empty_conn.close()
    
    def test_analytics_results_storage(self):
        """Test that analytics results are properly stored."""
        # Run analytics
        risky_objects = self.analytics.find_risky_objects(top_n=5)
        
        # Check that results are stored in analytics_results
        assert 'risky_objects' in self.analytics.analytics_results
        
        stored_results = self.analytics.analytics_results['risky_objects']
        
        # Should be list of dictionaries
        assert isinstance(stored_results, list)
        if stored_results:
            assert isinstance(stored_results[0], dict)
            assert len(stored_results) == len(risky_objects)
    
    def test_risk_category_assignment(self):
        """Test risk category assignment logic."""
        risky_df = self.analytics.find_risky_objects(top_n=10)
        
        if not risky_df.empty and 'risk_category' in risky_df.columns:
            valid_categories = ['VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW']
            
            # All categories should be valid
            assert all(cat in valid_categories for cat in risky_df['risk_category'])
            
            # Higher risk scores should generally have higher risk categories
            # (this is a loose check since categories depend on h and dist, not just risk_score)
            category_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'VERY_HIGH': 3}
            
            for _, row in risky_df.iterrows():
                assert row['risk_category'] in category_order
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'test_conn'):
            self.test_conn.close()
        
        if hasattr(self, 'analytics'):
            self.analytics.close()
        
        # Clean up any generated result files
        for result_file in Path('.').glob('analytics_results*.json'):
            try:
                result_file.unlink()
            except:
                pass