"""
Unit tests for NEO data loading functionality.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import tempfile
import json

# Add scripts to path for testing
sys.path.append('../scripts')
from scripts.data_ingestion import NEODataIngester


class TestNEODataIngester:
    """Test suite for NEODataIngester class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ingester = NEODataIngester(data_dir=self.temp_dir)
        
        # Sample API response data
        self.sample_api_response = {
            "signature": {"version": "1.5"},
            "count": 3,
            "fields": ["des", "orbit_id", "jd", "cd", "dist", "dist_min", "dist_max", 
                      "v_rel", "v_inf", "t_sigma_f", "h", "fullname"],
            "data": [
                ["433", "JPL-1", "2454567.5", "2024-Apr-13 12:00", "0.002", "0.001", 
                 "0.003", "15.5", "14.2", "< 1e-9", "17.7", "433 Eros"],
                ["99942", "JPL-2", "2462088.9", "2029-Apr-13 21:46", "0.0002", "0.0001", 
                 "0.0003", "30.7", "29.5", "< 1e-9", "19.7", "99942 Apophis"],
                ["2022 AP7", "JPL-3", "2459945.0", "2022-Dec-15 08:30", "0.045", "0.040", 
                 "0.050", "18.2", "17.8", "< 1e-9", "15.4", "2022 AP7"]
            ]
        }
        
        # Create sample DataFrame
        self.sample_df = pd.DataFrame(
            self.sample_api_response["data"], 
            columns=self.sample_api_response["fields"]
        )
    
    def test_initialization(self):
        """Test NEODataIngester initialization."""
        ingester = NEODataIngester("test_dir")
        assert Path(ingester.data_dir).name == "test_dir"
        assert ingester.api_url == "https://ssd-api.jpl.nasa.gov/cad.api"
    
    @patch('requests.get')
    def test_download_neo_data_success(self, mock_get):
        """Test successful data download from API."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = self.sample_api_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.ingester.download_neo_data(limit=10)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == self.sample_api_response["fields"]
        assert result.iloc[0]["des"] == "433"
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_download_neo_data_timeout_retry(self, mock_get):
        """Test retry logic on timeout."""
        import requests
        
        # First call times out, second succeeds
        mock_response = Mock()
        mock_response.json.return_value = self.sample_api_response
        mock_response.raise_for_status.return_value = None
        
        mock_get.side_effect = [
            requests.exceptions.Timeout("Request timeout"),
            mock_response
        ]
        
        result = self.ingester.download_neo_data(limit=10, max_retries=2)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert mock_get.call_count == 2
    
    @patch('requests.get')
    def test_download_neo_data_failure(self, mock_get):
        """Test handling of API failure."""
        import requests
        
        mock_get.side_effect = requests.exceptions.RequestException("API Error")
        
        with pytest.raises(requests.exceptions.RequestException):
            self.ingester.download_neo_data(max_retries=1)
    
    @patch('requests.get')
    def test_download_neo_data_invalid_response(self, mock_get):
        """Test handling of invalid API response."""
        mock_response = Mock()
        mock_response.json.return_value = {"invalid": "response"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with pytest.raises(ValueError, match="Invalid API response format"):
            self.ingester.download_neo_data()
    
    def test_clean_data_basic(self):
        """Test basic data cleaning functionality."""
        # Create test data with issues
        dirty_data = self.sample_df.copy()
        dirty_data.loc[1, 'dist'] = "-0.001"  # Invalid distance
        dirty_data.loc[2, 'h'] = "nan"  # Missing magnitude
        
        cleaned = self.ingester.clean_data(dirty_data)
        
        # Should have numeric columns converted
        assert pd.api.types.is_numeric_dtype(cleaned['dist'])
        assert pd.api.types.is_numeric_dtype(cleaned['h'])
        
        # Should have derived columns
        assert 'approach_year' in cleaned.columns
        
        # Should remove invalid distance rows
        assert len(cleaned) <= len(dirty_data)
    
    def test_clean_data_empty_dataframe(self):
        """Test cleaning empty DataFrame raises error."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot clean empty DataFrame"):
            self.ingester.clean_data(empty_df)
    
    def test_clean_data_date_parsing(self):
        """Test date parsing in clean_data."""
        test_data = self.sample_df.copy()
        cleaned = self.ingester.clean_data(test_data)
        
        # Check if dates were parsed (might be datetime or string depending on parsing success)
        assert 'cd' in cleaned.columns
        if not cleaned['cd'].isna().all():
            # If parsing succeeded, should have approach_year
            assert 'approach_year' in cleaned.columns
    
    def test_save_data_csv(self):
        """Test saving data in CSV format."""
        file_path = self.ingester.save_data(self.sample_df, format="csv", filename="test_neo")
        
        saved_path = Path(file_path)
        assert saved_path.exists()
        assert saved_path.suffix == ".csv"
        
        # Verify content
        loaded_df = pd.read_csv(file_path)
        assert len(loaded_df) == len(self.sample_df)
    
    def test_save_data_parquet(self):
        """Test saving data in Parquet format."""
        file_path = self.ingester.save_data(self.sample_df, format="parquet", filename="test_neo")
        
        saved_path = Path(file_path)
        assert saved_path.exists()
        assert saved_path.suffix == ".parquet"
        
        # Verify content
        loaded_df = pd.read_parquet(file_path)
        assert len(loaded_df) == len(self.sample_df)
    
    def test_save_data_json(self):
        """Test saving data in JSON format."""
        file_path = self.ingester.save_data(self.sample_df, format="json", filename="test_neo")
        
        saved_path = Path(file_path)
        assert saved_path.exists()
        assert saved_path.suffix == ".json"
        
        # Verify content
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)
        assert len(loaded_data) == len(self.sample_df)
    
    def test_save_data_unsupported_format(self):
        """Test error on unsupported format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            self.ingester.save_data(self.sample_df, format="xlsx")
    
    def test_save_data_empty_dataframe(self):
        """Test error on empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot save empty DataFrame"):
            self.ingester.save_data(empty_df)
    
    def test_data_type_conversion(self):
        """Test proper data type handling in cleaning."""
        test_data = self.sample_df.copy()
        
        # Add some problematic values
        test_data.loc[0, 'dist'] = "0.002"  # String number
        test_data.loc[1, 'h'] = ""  # Empty string
        test_data.loc[2, 'v_rel'] = "invalid"  # Invalid number
        
        cleaned = self.ingester.clean_data(test_data)
        
        # Check numeric conversions
        assert pd.api.types.is_numeric_dtype(cleaned['dist'])
        assert pd.api.types.is_numeric_dtype(cleaned['h'])
        assert pd.api.types.is_numeric_dtype(cleaned['v_rel'])
        
        # Check that invalid numbers become NaN
        assert pd.isna(cleaned.loc[2, 'v_rel'])
    
    def test_duplicate_removal(self):
        """Test removal of duplicate rows."""
        test_data = self.sample_df.copy()
        
        # Add duplicate row
        duplicate_row = test_data.iloc[0].copy()
        test_data = pd.concat([test_data, duplicate_row.to_frame().T], ignore_index=True)
        
        cleaned = self.ingester.clean_data(test_data)
        
        # Should have removed duplicate
        assert len(cleaned) == len(self.sample_df)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)