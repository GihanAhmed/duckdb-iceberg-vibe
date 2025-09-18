"""
DuckDB Setup & Operations for Space Analytics Demo

This script sets up DuckDB databases, creates optimized tables for NEO data,
and provides utilities for querying and managing the data.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import duckdb

# Try to import config, fallback if not available
try:
    from config.duckdb_config import duckdb_config
except ImportError:
    # Fallback configuration
    class MockConfig:
        """Mock configuration class for when config is unavailable."""
        @staticmethod
        def get_connection_config():
            """Get connection configuration."""
            return {}
        @staticmethod
        def get_extensions():
            """Get list of extensions."""
            return ['parquet', 'json']
    duckdb_config = MockConfig()

sys.path.append('..')


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DuckDBManager:
    """
    Manages DuckDB databases for the Space Analytics demo.

    Provides functionality for creating optimized tables, loading data,
    setting up MotherDuck synchronization, and executing analytics queries.
    """

    def __init__(self, db_path: str = "space_analytics.db"):
        """Initialize DuckDB manager with database connection."""
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._setup_extensions()
    def _connect(self) -> None:
        """Establish DuckDB connection with optimized settings."""
        try:
            logger.info("Connecting to DuckDB database: %s", self.db_path)
            config = duckdb_config.get_connection_config()
            self.conn = duckdb.connect(self.db_path, **config)

            # Set additional performance optimizations
            self.conn.execute("SET memory_limit='4GB'")
            self.conn.execute("SET threads=4")
            self.conn.execute("SET enable_progress_bar=true")

            logger.info("✓ DuckDB connection established successfully")

        except (duckdb.Error, OSError) as exc:
            logger.error("Failed to connect to DuckDB: %s", exc)
            raise

    def _setup_extensions(self) -> None:
        """Install and load required DuckDB extensions."""
        extensions = duckdb_config.get_extensions()
        for ext in extensions:
            try:
                logger.info("Installing extension: %s", ext)
                self.conn.execute(f"INSTALL {ext}")
                self.conn.execute(f"LOAD {ext}")
                logger.info("✓ Extension %s loaded successfully", ext)
            except duckdb.Error as exc:
                logger.warning("Failed to load extension %s: %s", ext, exc)

    def create_neo_table(self, data_path: str) -> None:
        """
        Create optimized NEO table from data file.

        Args:
            data_path: Path to the NEO data file (CSV, Parquet, or JSON)

        Raises:
            FileNotFoundError: If data file doesn't exist
            Exception: If table creation fails
        """
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info("Creating NEO table from %s", data_path)

        try:
            # Drop existing table if it exists
            self.conn.execute("DROP TABLE IF EXISTS neo_approaches")

            # Detect file format and create table accordingly
            file_ext = Path(data_path).suffix.lower()

            if file_ext == '.csv':
                create_sql = f"""
                CREATE TABLE neo_approaches AS
                SELECT * FROM read_csv_auto('{data_path}')
                """
            elif file_ext == '.parquet':
                create_sql = f"""
                CREATE TABLE neo_approaches AS
                SELECT * FROM '{data_path}'
                """
            elif file_ext == '.json':
                create_sql = f"""
                CREATE TABLE neo_approaches AS
                SELECT * FROM read_json_auto('{data_path}')
                """
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            self.conn.execute(create_sql)

            # Create indexes for better performance
            index_queries = [
                "CREATE INDEX idx_neo_des ON neo_approaches(des)",
                "CREATE INDEX idx_neo_cd ON neo_approaches(cd)",
                "CREATE INDEX idx_neo_dist ON neo_approaches(dist)",
                "CREATE INDEX idx_neo_h ON neo_approaches(h)",
                "CREATE INDEX idx_neo_year ON neo_approaches(approach_year)"
            ]

            for query in index_queries:
                try:
                    self.conn.execute(query)
                except duckdb.Error as exc:
                    logger.warning("Index creation warning: %s", exc)

            # Get table statistics
            stats = self.get_table_stats("neo_approaches")
            logger.info("✓ NEO table created successfully:")
            logger.info("  - Rows: %s", f"{stats['row_count']:,}")
            logger.info("  - Columns: %d", stats['column_count'])
            logger.info("  - Storage size: %s", stats.get('storage_size', 'Unknown'))

        except (duckdb.Error, FileNotFoundError, ValueError) as exc:
            logger.error("Failed to create NEO table: %s", exc)
            raise


    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute query and return results as list of dictionaries.

        Args:
            query: SQL query to execute

        Returns:
            Query results as list of dictionaries
        """
        try:
            result = self.conn.execute(query)

            # Get column names
            columns = [desc[0] for desc in result.description]

            # Fetch all rows and convert to dictionaries
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]

        except duckdb.Error as exc:
            logger.error("Query execution failed: %s", exc)
            raise

    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive table statistics.

        Args:
            table_name: Name of the table to analyze

        Returns:
            Dictionary containing table statistics
        """
        try:
            stats = {}

            # Row count
            row_count = self.conn.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]
            stats['row_count'] = row_count

            # Column information
            columns_info = self.conn.execute(
                f"DESCRIBE {table_name}"
            ).fetchall()
            stats['column_count'] = len(columns_info)
            stats['columns'] = [
                {'name': col[0], 'type': col[1]} for col in columns_info
            ]

            # Date range (if cd column exists)
            try:
                date_range = self.conn.execute(f"""
                    SELECT MIN(cd) as min_date, MAX(cd) as max_date
                    FROM {table_name}
                    WHERE cd IS NOT NULL
                """).fetchone()
                if date_range[0] and date_range[1]:
                    stats['date_range'] = {
                        'min': date_range[0], 'max': date_range[1]
                    }
            except Exception:  # pylint: disable=broad-except
                pass

            # Storage size estimation
            try:
                size_info = self.conn.execute(
                    f"SELECT pg_total_relation_size('{table_name}') as size_bytes"
                ).fetchone()
                if size_info[0]:
                    stats['storage_size'] = f"{size_info[0] / (1024*1024):.2f} MB"
            except Exception:  # pylint: disable=broad-except
                stats['storage_size'] = "Unknown"

            return stats

        except duckdb.Error as exc:
            logger.error("Failed to get table stats: %s", exc)
            return {'error': str(exc)}

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("✓ DuckDB connection closed")


# Sample queries for demonstration
SAMPLE_QUERIES = {
    "total_count": "SELECT COUNT(*) as total_approaches FROM neo_approaches",

    "dangerous_objects": """
        SELECT des, fullname, h, dist, v_rel
        FROM neo_approaches
        WHERE h < 18 AND dist < 0.05
        ORDER BY dist ASC
        LIMIT 10
    """,

    "yearly_discoveries": """
        SELECT
            approach_year as year,
            COUNT(*) as approaches,
            ROUND(AVG(dist), 6) as avg_distance_au,
            ROUND(MIN(dist), 6) as closest_approach_au,
            COUNT(DISTINCT des) as unique_objects
        FROM neo_approaches
        WHERE approach_year IS NOT NULL
        GROUP BY approach_year
        ORDER BY approach_year DESC
        LIMIT 20
    """,

    "size_distribution": """
        SELECT
            CASE
                WHEN h < 18 THEN 'Very Large (>1km)'
                WHEN h < 22 THEN 'Large (140m-1km)'
                WHEN h < 25 THEN 'Medium (30-140m)'
                ELSE 'Small (<30m)'
            END as size_category,
            COUNT(*) as count,
            ROUND(MIN(dist), 6) as closest_ever_au,
            ROUND(AVG(v_rel), 2) as avg_velocity_kms
        FROM neo_approaches
        WHERE h IS NOT NULL
        GROUP BY size_category
        ORDER BY MIN(h)
    """,

    "recent_close_approaches": """
        SELECT
            des,
            fullname,
            cd as approach_date,
            ROUND(dist, 6) as distance_au,
            ROUND(v_rel, 2) as velocity_kms,
            h as magnitude
        FROM neo_approaches
        WHERE cd >= '2020-01-01' AND dist < 0.01
        ORDER BY dist ASC
        LIMIT 15
    """
}


def main():
    """Main execution function for DuckDB setup."""
    logger.info("=" * 60)
    logger.info("DUCKDB SETUP & NEO DATA LOADING")
    logger.info("=" * 60)
    try:
        # Initialize DuckDB manager
        db_manager = DuckDBManager()

        # Look for data files
        data_dir = Path("data/raw")
        data_files = (
            list(data_dir.glob("neo_data_*.csv")) +
            list(data_dir.glob("neo_data_*.parquet"))
        )

        if not data_files:
            logger.error("No NEO data files found in data/raw directory")
            logger.info("Please run: python scripts/02_data_ingestion.py first")
            return False

        # Use the most recent data file
        data_file = max(data_files, key=lambda p: p.stat().st_mtime)
        logger.info("Using data file: %s", data_file)

        # Create NEO table
        db_manager.create_neo_table(str(data_file))

        # Execute sample queries
        logger.info("\n%s", "=" * 60)
        logger.info("EXECUTING SAMPLE QUERIES")
        logger.info("=" * 60)

        for query_name, query in SAMPLE_QUERIES.items():
            try:
                logger.info(
                    "\n--- %s ---", query_name.upper().replace('_', ' ')
                )
                results = db_manager.execute_query(query)

                if results:
                    # Print first few results
                    for i, row in enumerate(results[:5]):
                        logger.info("Row %d: %s", i + 1, row)

                    if len(results) > 5:
                        logger.info(
                            "... and %d more rows", len(results) - 5
                        )
                else:
                    logger.info("No results returned")

            except duckdb.Error as exc:
                logger.error("Query '%s' failed: %s", query_name, exc)


        # Final summary
        stats = db_manager.get_table_stats("neo_approaches")
        logger.info("\n%s", "=" * 60)
        logger.info("SETUP SUMMARY")
        logger.info("=" * 60)
        logger.info("Database: %s", db_manager.db_path)
        logger.info("Total records: %s", f"{stats['row_count']:,}")
        logger.info("Columns: %d", stats['column_count'])
        if 'date_range' in stats:
            logger.info(
                "Date range: %s to %s",
                stats['date_range']['min'],
                stats['date_range']['max']
            )

        logger.info("\n🎉 DuckDB setup completed successfully!")
        logger.info("Next step: python scripts/04_iceberg_conversion.py")

        return True

    except (duckdb.Error, FileNotFoundError, OSError) as exc:
        logger.error("❌ DuckDB setup failed: %s", exc)
        return False
    finally:
        if 'db_manager' in locals():
            db_manager.close()


if __name__ == "__main__":
    SUCCESS = main()
    sys.exit(0 if SUCCESS else 1)
