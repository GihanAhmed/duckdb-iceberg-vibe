"""
NASA NEO Data Ingestion Pipeline

This script downloads Near-Earth Object close approach data from NASA's JPL API,
cleans and validates the data, and saves it in various formats for analysis.
"""
import logging
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
import requests

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

    def download_neo_data(self, limit: int = 50000, max_retries: int = 3) -> pd.DataFrame:
        """
        Download NEO close approach data from NASA API with retry logic.

        Args:
            limit: Maximum number of records to download
            max_retries: Maximum number of retry attempts for failed requests

        Returns:
            DataFrame containing NEO close approach data

        Raises:
            requests.RequestException: If API request fails after all retries
            ValueError: If API returns invalid or empty data
        """
        logger.info("Downloading NEO data from NASA JPL API (limit: %d)", limit)

        params = {
            'date-min': '1900-01-01',
            'date-max': '2030-12-31',
            'dist-max': '0.2',  # Within 0.2 AU
            'sort': 'date',
            'limit': str(limit),
            'fullname': 'true'
        }

        for attempt in range(max_retries):
            try:
                logger.info("API request attempt %d/%d", attempt + 1, max_retries)

                response = requests.get(
                    self.api_url,
                    params=params,
                    timeout=60,
                    headers={'User-Agent': 'SpaceAnalytics/1.0'}
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


def main():
    """Main execution function for data ingestion."""
    logger.info("=" * 60)
    logger.info("NASA NEO DATA INGESTION PIPELINE")
    logger.info("=" * 60)

    try:
        # Initialize ingester
        ingester = NEODataIngester()

        # Download data
        raw_data = ingester.download_neo_data(limit=50000)

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
