"""
NASA NEO Data Ingestion Pipeline

This script downloads Near-Earth Object close approach data from NASA's JPL API,
cleans and validates the data, and saves it in various formats for analysis.
"""
import logging
import time
import psutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import math

import pandas as pd
import requests
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NEODataIngester:
    """
    Handles downloading and processing NASA JPL Close Approach Data.

    This class provides a simple interface to fetch Near-Earth Object data
    from NASA's JPL API and convert it to various formats for analysis.

    Attributes:
        data_dir (Path): Directory to store raw data files
        api_url (str): Base URL for NASA JPL CAD API
    """

    def __init__(self, data_dir: str = "data/raw"):
        """Initialize the data loader with directory and API settings."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.api_url = "https://ssd-api.jpl.nasa.gov/cad.api"
        self.memory_threshold_gb = 8.0  # Memory usage threshold for sampling strategies
        self.chunk_size = 10000  # Records per API chunk to avoid rate limits

    def download_neo_data(
        self,
        full_dataset: bool = True,
        limit: Optional[int] = None,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Download NEO close approach data from NASA API with comprehensive dataset support.

        Args:
            full_dataset: If True, download complete dataset ignoring limit
            limit: Maximum number of records (only used if full_dataset=False)
            max_retries: Maximum number of retry attempts for failed requests

        Returns:
            DataFrame containing NEO close approach data

        Raises:
            requests.RequestException: If API request fails after all retries
            ValueError: If API returns invalid or empty data
            MemoryError: If dataset exceeds memory threshold without sampling
        """
        if full_dataset:
            logger.info("Downloading COMPLETE NEO dataset from NASA JPL API")
            return self._download_full_dataset(max_retries)
        else:
            limit = limit or 50000
            logger.info("Downloading NEO data from NASA JPL API (limit: %d)", limit)
            return self._download_limited_dataset(limit, max_retries)

    def _download_full_dataset(self, max_retries: int = 3) -> pd.DataFrame:
        """
        Download complete dataset using pagination and memory management.

        Returns:
            Complete DataFrame with all available NEO data
        """
        base_params = {
            'date-min': '1900-01-01',
            'date-max': '2100-12-31',
            'dist-max': '10',  # Extended to 10 AU for comprehensive data
            'sort': 'date',
            'fullname': 'true',
            'diameter': 'true'  # Include diameter information
        }

        all_data = []
        offset = 0
        total_records = 0

        logger.info("Starting comprehensive dataset download with chunking...")

        while True:
            # Monitor memory usage
            memory_usage_gb = psutil.virtual_memory().used / (1024**3)
            if memory_usage_gb > self.memory_threshold_gb:
                logger.warning("Memory usage (%.1f GB) exceeds threshold (%.1f GB)",
                              memory_usage_gb, self.memory_threshold_gb)
                logger.info("Implementing data sampling strategy...")
                break

            chunk_params = base_params.copy()
            chunk_params.update({
                'limit': str(self.chunk_size),
                'offset': str(offset)
            })

            logger.info("Fetching chunk: offset=%d, size=%d", offset, self.chunk_size)

            chunk_df = self._fetch_api_chunk(chunk_params, max_retries)

            if chunk_df.empty:
                logger.info("No more data available. Download complete.")
                break

            all_data.append(chunk_df)
            total_records += len(chunk_df)
            offset += self.chunk_size

            logger.info("Downloaded %d total records so far...", total_records)

            # If chunk is smaller than requested, we've reached the end
            if len(chunk_df) < self.chunk_size:
                logger.info("Reached end of dataset. Download complete.")
                break

            # Rate limiting - be respectful to NASA's API
            time.sleep(1)

        if not all_data:
            raise ValueError("No data retrieved from NASA API")

        logger.info("Concatenating %d chunks with %d total records", len(all_data), total_records)
        full_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates that might occur at chunk boundaries
        initial_count = len(full_df)
        full_df = full_df.drop_duplicates(subset=['des', 'cd'], keep='first')
        final_count = len(full_df)

        if initial_count != final_count:
            logger.info("Removed %d duplicate records", initial_count - final_count)

        logger.info("✅ Complete dataset download finished: %d unique records", final_count)
        return full_df

    def _download_limited_dataset(self, limit: int, max_retries: int = 3) -> pd.DataFrame:
        """
        Download limited dataset (legacy method for compatibility).
        """
        params = {
            'date-min': '1900-01-01',
            'date-max': '2030-12-31',
            'dist-max': '0.2',  # Within 0.2 AU
            'sort': 'date',
            'limit': str(limit),
            'fullname': 'true'
        }

        return self._fetch_api_chunk(params, max_retries)

    def _fetch_api_chunk(self, params: Dict[str, str], max_retries: int) -> pd.DataFrame:
        """
        Fetch a single chunk of data from the NASA API.
        """

        for attempt in range(max_retries):
            try:
                logger.debug("API chunk request attempt %d/%d", attempt + 1, max_retries)

                response = requests.get(
                    self.api_url,
                    params=params,
                    timeout=120,  # Increased timeout for larger datasets
                    headers={'User-Agent': 'SpaceAnalytics/2.0'}
                )
                response.raise_for_status()

                data = response.json()

                # Validate API response structure
                if 'data' not in data or 'fields' not in data:
                    raise ValueError("Invalid API response format")

                if not data['data']:
                    raise ValueError("API returned empty dataset")

                logger.info("Successfully downloaded %d records", len(data['data']))

                # Convert to DataFrame
                df_result = pd.DataFrame(data['data'], columns=data['fields'])
                logger.info("Created DataFrame with shape: %s", df_result.shape)

                return df_result

            except requests.exceptions.Timeout:
                logger.warning("Request timeout on attempt %d", attempt + 1)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

            except requests.exceptions.RequestException as exc:
                logger.error("Request failed on attempt %d: %s", attempt + 1, exc)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

            except (ValueError, KeyError) as exc:
                logger.error("Data validation error: %s", exc)
                raise

        raise requests.RequestException(
            f"Failed to download data after {max_retries} attempts"
        )

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate NEO data.

        Args:
            df: Raw DataFrame from NASA API

        Returns:
            Cleaned and validated DataFrame

        Raises:
            ValueError: If data cleaning fails
        """
        logger.info("Cleaning and validating NEO data...")

        if df.empty:
            raise ValueError("Cannot clean empty DataFrame")

        # Create a copy to avoid modifying original
        cleaned_df = df.copy()

        # Clean column names (handle potential spacing issues)
        cleaned_df.columns = [col.strip() for col in cleaned_df.columns]

        # Parse and validate calendar date
        try:
            cleaned_df['cd'] = pd.to_datetime(
                cleaned_df['cd'], format='%Y-%b-%d %H:%M', errors='coerce'
            )
            invalid_dates = cleaned_df['cd'].isna().sum()
            if invalid_dates > 0:
                logger.warning("Found %d invalid dates, setting to NaT", invalid_dates)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Date parsing failed: %s, keeping original format", exc)

        # Convert numeric columns
        numeric_columns = ['jd', 'dist', 'dist_min', 'dist_max', 'v_rel', 'v_inf', 'h']

        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

        # Validate distance values (must be positive)
        if 'dist' in cleaned_df.columns:
            invalid_dist = (cleaned_df['dist'] <= 0) | cleaned_df['dist'].isna()
            invalid_count = invalid_dist.sum()
            if invalid_count > 0:
                logger.warning("Found %d invalid distance values", invalid_count)
                cleaned_df = cleaned_df[~invalid_dist]

        # Clean object names and designations
        text_columns = ['des', 'fullname', 't_sigma_f']
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                cleaned_df[col] = cleaned_df[col].replace(['nan', 'None', ''], pd.NA)

        # Add derived columns
        if 'cd' in cleaned_df.columns and not cleaned_df['cd'].isna().all():
            cleaned_df['approach_year'] = cleaned_df['cd'].dt.year
        else:
            # Fallback: extract year from string format if datetime parsing failed
            try:
                cleaned_df['approach_year'] = cleaned_df['cd'].str[:4].astype(int)
            except (ValueError, TypeError, AttributeError):
                logger.warning("Could not extract approach year")
                cleaned_df['approach_year'] = None

        # Remove duplicates
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_df)
        if removed_duplicates > 0:
            logger.info("Removed %d duplicate rows", removed_duplicates)

        # Final validation
        if cleaned_df.empty:
            raise ValueError("All data was filtered out during cleaning")

        logger.info("Data cleaning completed. Final shape: %s", cleaned_df.shape)
        logger.info("Columns: %s", list(cleaned_df.columns))

        # Log data quality summary
        logger.info("Data quality summary:")
        for col in cleaned_df.columns:
            null_count = cleaned_df[col].isna().sum()
            null_pct = (null_count / len(cleaned_df)) * 100
            logger.info("  %s: %d nulls (%.1f%%)", col, null_count, null_pct)

        return cleaned_df

    def save_data(self, df: pd.DataFrame, output_format: str = "csv",
                  filename: Optional[str] = None) -> str:
        """
        Save data in specified format.

        Args:
            df: DataFrame to save
            output_format: Output format ('csv', 'parquet', 'json')
            filename: Optional custom filename

        Returns:
            Path to saved file

        Raises:
            ValueError: If unsupported format specified
        """
        if df.empty:
            raise ValueError("Cannot save empty DataFrame")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if filename is None:
            filename = f"neo_data_{timestamp}"

        supported_formats = ["csv", "parquet", "json"]
        if output_format not in supported_formats:
            raise ValueError(
                f"Unsupported format: {output_format}. Use one of: {supported_formats}"
            )

        file_path = self.data_dir / f"{filename}.{output_format}"

        logger.info("Saving data to %s in %s format...", file_path, output_format)

        try:
            if output_format == "csv":
                df.to_csv(file_path, index=False)
            elif output_format == "parquet":
                df.to_parquet(file_path, index=False)
            elif output_format == "json":
                df.to_json(file_path, orient='records', date_format='iso')

            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            logger.info("Successfully saved %d records to %s (%.2f MB)",
                       len(df), file_path, file_size)

            return str(file_path)

        except (OSError, IOError, ValueError) as exc:
            logger.error("Failed to save data: %s", exc)
            raise

    def get_memory_usage_info(self) -> Dict[str, float]:
        """
        Get current memory usage information.

        Returns:
            Dictionary with memory usage statistics in GB
        """
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent
        }

    def implement_sampling_strategy(
        self,
        df: pd.DataFrame,
        target_size: int = 100000,
        strategy: str = 'stratified'
    ) -> pd.DataFrame:
        """
        Implement data sampling strategy for large datasets.

        Args:
            df: Input DataFrame
            target_size: Target number of records after sampling
            strategy: Sampling strategy ('random', 'stratified', 'temporal')

        Returns:
            Sampled DataFrame
        """
        if len(df) <= target_size:
            return df

        logger.info("Implementing %s sampling: %d -> %d records",
                   strategy, len(df), target_size)

        if strategy == 'random':
            return df.sample(n=target_size, random_state=42)

        elif strategy == 'stratified':
            # Stratified sampling by risk category if available
            if 'risk_category' in df.columns:
                return df.groupby('risk_category').apply(
                    lambda x: x.sample(n=min(len(x), target_size // 4), random_state=42)
                ).reset_index(drop=True)
            else:
                # Fallback to temporal stratification
                return self.implement_sampling_strategy(df, target_size, 'temporal')

        elif strategy == 'temporal':
            # Sample evenly across time periods
            if 'cd' in df.columns:
                df['year'] = pd.to_datetime(df['cd']).dt.year
                years = sorted(df['year'].unique())
                samples_per_year = target_size // len(years)

                sampled_dfs = []
                for year in years:
                    year_data = df[df['year'] == year]
                    sample_size = min(len(year_data), samples_per_year)
                    if sample_size > 0:
                        sampled_dfs.append(year_data.sample(n=sample_size, random_state=42))

                result = pd.concat(sampled_dfs, ignore_index=True)
                return result.drop('year', axis=1)
            else:
                return df.sample(n=target_size, random_state=42)

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")


def main():
    """Main execution function for data ingestion."""
    logger.info("=" * 60)
    logger.info("NASA NEO DATA INGESTION PIPELINE")
    logger.info("=" * 60)

    try:
        # Initialize ingester
        ingester = NEODataIngester()

        # Download complete dataset
        logger.info("Starting comprehensive dataset download...")
        memory_info = ingester.get_memory_usage_info()
        logger.info("Initial memory usage: %.1f GB (%.1f%% used)",
                   memory_info['used_gb'], memory_info['percent_used'])

        raw_data = ingester.download_neo_data(full_dataset=True)

        # Clean data
        cleaned_data = ingester.clean_data(raw_data)

        # Save in multiple formats
        formats = ["csv", "parquet", "json"]
        saved_files = []

        for fmt in formats:
            try:
                file_path = ingester.save_data(cleaned_data, output_format=fmt)
                saved_files.append(file_path)
            except (OSError, IOError, ValueError) as exc:
                logger.error("Failed to save %s format: %s", fmt, exc)

        # Summary
        logger.info("\n%s", "=" * 60)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info("Records downloaded: %d", len(raw_data))
        logger.info("Records after cleaning: %d", len(cleaned_data))
        logger.info("Files saved: %d", len(saved_files))

        for file_path in saved_files:
            logger.info("  ✓ %s", file_path)

        logger.info("\n🎉 Data ingestion completed successfully!")
        logger.info("Next step: python scripts/03_duckdb_setup.py")

        return True

    except (requests.RequestException, ValueError, OSError, IOError) as exc:
        logger.error("❌ Data ingestion failed: %s", exc)
        return False


if __name__ == "__main__":
    import sys
    SUCCESS_RESULT = main()
    sys.exit(0 if SUCCESS_RESULT else 1)
