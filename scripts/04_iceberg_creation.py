"""
Apache Iceberg Table Creation

Creates true Iceberg tables from parquet source with time travel and schema evolution.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.partitioning import PartitionSpec
from pyiceberg.types import (
    NestedField,
    StringType,
    TimestampType,
    DoubleType,
    LongType
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IcebergManager:
    """Create and manage Apache Iceberg tables using PyIceberg."""

    def __init__(self, warehouse_path: str = "data/iceberg_warehouse"):
        """Initialize Iceberg manager with warehouse path.

        Args:
            warehouse_path: Path to Iceberg warehouse directory
        """
        self.warehouse_path = Path(warehouse_path)
        self.warehouse_path.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite catalog
        try:
            db_path = self.warehouse_path / "catalog.db"
            self.catalog = SqlCatalog(
                "demo_catalog",
                **{
                    "uri": f"sqlite:///{db_path}",
                    "warehouse": f"file://{self.warehouse_path.absolute()}"
                }
            )
            logger.info("✓ PyIceberg SQLite catalog initialized")
        except Exception as e:
            logger.error("Failed to initialize catalog: %s", e)
            raise

        # Path to source parquet file - find the latest one
        parquet_files = sorted(Path("data/raw").glob("neo_data_*.parquet"))
        if parquet_files:
            self.source_parquet = parquet_files[-1]
        else:
            self.source_parquet = Path("data/raw/neo_data_20250917_121532.parquet")
        if not self.source_parquet.exists():
            raise FileNotFoundError(
                f"Source parquet file not found: {self.source_parquet}"
            )

    def create_iceberg_table(self) -> str:
        """Create Iceberg table from parquet source.

        Returns:
            Table identifier string
        """
        table_name = "neo_approaches_iceberg"

        try:
            logger.info(
                "Creating Iceberg table '%s' from parquet file",
                table_name
            )

            # Read parquet file
            df = pd.read_parquet(self.source_parquet)
            logger.info(
                "Source parquet has %d columns and %d records",
                len(df.columns),
                len(df)
            )

            # Add approach_year column if not present
            if 'approach_year' not in df.columns:
                df['approach_year'] = pd.to_datetime(df['cd']).dt.year

            # Convert cd to string format to avoid nanosecond precision issues
            df['cd'] = pd.to_datetime(df['cd']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Define Iceberg schema based on parquet columns
            schema = Schema(
                NestedField(1, "des", StringType(), required=False),
                NestedField(2, "fullname", StringType(), required=False),
                NestedField(3, "cd", StringType(), required=False),  # Store as string
                NestedField(4, "dist", DoubleType(), required=False),
                NestedField(5, "dist_min", DoubleType(), required=False),
                NestedField(6, "dist_max", DoubleType(), required=False),
                NestedField(7, "v_rel", DoubleType(), required=False),
                NestedField(8, "v_inf", DoubleType(), required=False),
                NestedField(9, "h", DoubleType(), required=False),
                NestedField(10, "orbit_id", StringType(), required=False),
                NestedField(11, "t_sigma_f", StringType(), required=False),
                NestedField(12, "jd", DoubleType(), required=False),
                NestedField(13, "approach_year", LongType(), required=False)
            )

            # Create namespace
            namespace = "demo"
            try:
                self.catalog.create_namespace(namespace)
                logger.info("✓ Created namespace '%s'", namespace)
            except Exception as e:
                logger.info("Namespace may already exist: %s", e)

            # Create or load table
            table_identifier = f"{namespace}.{table_name}"
            try:
                # Create empty partition spec (no partitioning)
                partition_spec = PartitionSpec()

                table = self.catalog.create_table(
                    table_identifier,
                    schema=schema,
                    partition_spec=partition_spec
                )
                logger.info("✓ Created new Iceberg table")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info("Table already exists, loading existing table")
                    table = self.catalog.load_table(table_identifier)
                else:
                    raise

            # Convert to PyArrow table and write
            arrow_table = pa.Table.from_pandas(df)
            table.append(arrow_table)
            logger.info("✓ Wrote %d records to Iceberg table", len(df))

            return table_identifier

        except Exception as e:
            logger.error("Failed to create Iceberg table: %s", e)
            raise

    def add_simple_column(self, table_name: str) -> Dict[str, Any]:
        """Add a simple column for schema evolution demo.

        Args:
            table_name: Table identifier

        Returns:
            Operation results dictionary
        """
        try:
            logger.info("Demonstrating schema evolution")

            table = self.catalog.load_table(table_name)

            # Add new column
            with table.update_schema() as update:
                update.add_column("risk_level", StringType())

            logger.info("✓ Added 'risk_level' column to schema")

            # Add sample data with new column
            new_data = pd.DataFrame({
                'des': ['DEMO1', 'DEMO2'],
                'fullname': ['Demo Object 1', 'Demo Object 2'],
                'cd': ['2024-01-01 12:00:00', '2024-01-01 12:00:00'],  # String format
                'dist': [0.01, 0.02],
                'dist_min': [0.008, 0.018],
                'dist_max': [0.012, 0.022],
                'v_rel': [10.5, 15.2],
                'v_inf': [8.1, 12.3],
                'h': [20.1, 19.8],
                'orbit_id': ['demo_1', 'demo_2'],
                't_sigma_f': ['< 1 day', '< 1 day'],
                'jd': [2460000.0, 2460001.0],
                'approach_year': [2024, 2024],
                'risk_level': ['medium', 'low']
            })

            # Convert and append
            new_arrow_table = pa.Table.from_pandas(new_data)
            table.append(new_arrow_table)
            logger.info("✓ Added sample records with new column")

            return {
                "operation": "add_column",
                "column_name": "risk_level",
                "column_type": "string",
                "records_added": len(new_data),
                "status": "completed"
            }

        except Exception as e:
            logger.error("Schema evolution failed: %s", e)
            return {"error": str(e)}

    def demonstrate_time_travel(self, table_name: str) -> Dict[str, Any]:
        """Query different snapshots to show time travel.

        Args:
            table_name: Table identifier

        Returns:
            Time travel results dictionary
        """
        try:
            logger.info("Demonstrating time travel capabilities")

            table = self.catalog.load_table(table_name)

            # Get snapshots
            snapshots = list(table.history())
            logger.info("Found %d snapshots", len(snapshots))

            results = {
                "snapshots": [],
                "queries": []
            }

            # Process each snapshot
            for i, snapshot in enumerate(snapshots):
                snap_info = {
                    "snapshot_id": snapshot.snapshot_id,
                    "timestamp": datetime.fromtimestamp(
                        snapshot.timestamp_ms / 1000
                    ).isoformat(),
                    "summary": dict(snapshot.summary) if hasattr(
                        snapshot,
                        'summary'
                    ) else {}
                }
                results["snapshots"].append(snap_info)

                # Query this snapshot
                try:
                    scan = table.scan(snapshot_id=snapshot.snapshot_id)
                    arrow_table = scan.to_arrow()
                    record_count = len(arrow_table)

                    query_result = {
                        "snapshot_id": snapshot.snapshot_id,
                        "timestamp": snap_info["timestamp"],
                        "record_count": record_count
                    }
                    results["queries"].append(query_result)

                    logger.info(
                        "Snapshot %d: %d records at %s",
                        i + 1,
                        record_count,
                        snap_info['timestamp']
                    )

                except Exception as e:
                    logger.warning(
                        "Could not query snapshot %s: %s",
                        snapshot.snapshot_id,
                        e
                    )

            logger.info("✓ Time travel demonstration completed")
            return results

        except Exception as e:
            logger.error("Time travel demonstration failed: %s", e)
            return {"error": str(e)}

    def close(self) -> None:
        """Close connections."""
        logger.info("✓ Catalog closed")


def save_results_to_markdown(results: Dict[str, Any]) -> None:
    """Save results to markdown file.

    Args:
        results: Dictionary containing demo results
    """
    output_file = "iceberg_features_demo.md"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 🧊 Apache Iceberg Features Demo\n\n")
        f.write("This document showcases time travel and schema evolution ")
        f.write("capabilities of Apache Iceberg tables created with PyIceberg.\n\n")

        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Table creation
        f.write("## 📊 Table Creation\n\n")
        f.write(f"- **Table Name:** `{results['table_name']}`\n")
        f.write(f"- **Records:** {results.get('record_count', 'N/A'):,}\n")
        f.write("- **Format:** Apache Iceberg\n")
        f.write("- **Catalog:** SQLite-based\n")
        f.write("- **Partitioning:** None (as requested)\n\n")

        # Schema evolution
        if 'schema_evolution' in results:
            evolution = results['schema_evolution']
            f.write("## 🔄 Schema Evolution\n\n")
            if 'error' not in evolution:
                f.write(
                    f"✅ **Successfully added column:** "
                    f"`{evolution['column_name']}` ({evolution['column_type']})\n"
                )
                f.write(f"- Records added with new column: {evolution['records_added']}\n")
                f.write("- Backward compatibility maintained\n")
                f.write("- No data migration required\n\n")
            else:
                f.write(f"❌ **Schema evolution failed:** {evolution['error']}\n\n")

        # Time travel
        if 'time_travel' in results:
            time_travel = results['time_travel']
            f.write("## ⏰ Time Travel\n\n")

            if 'error' not in time_travel:
                f.write(
                    f"Found **{len(time_travel['snapshots'])} snapshots** "
                    f"in table history:\n\n"
                )

                for i, snapshot in enumerate(time_travel['snapshots']):
                    f.write(f"### Snapshot {i+1}\n")
                    f.write(f"- **ID:** `{snapshot['snapshot_id']}`\n")
                    f.write(f"- **Timestamp:** {snapshot['timestamp']}\n")
                    if snapshot.get('summary'):
                        f.write(f"- **Summary:** {snapshot['summary']}\n")
                    f.write("\n")

                f.write("### Query Results by Snapshot\n\n")
                f.write("| Snapshot | Timestamp | Records |\n")
                f.write("|----------|-----------|----------|\n")

                for query in time_travel['queries']:
                    timestamp = query['timestamp'][:19]
                    snapshot_id = str(query['snapshot_id'])
                    f.write(
                        f"| `{snapshot_id[:8]}...` | {timestamp} | "
                        f"{query['record_count']:,} |\n"
                    )
                f.write("\n")
            else:
                f.write(f"❌ **Time travel failed:** {time_travel['error']}\n\n")

        f.write("## 🎯 Key Benefits Demonstrated\n\n")
        f.write("✅ **True Iceberg Format:** Proper metadata management with JSON files\n")
        f.write("✅ **ACID Transactions:** Safe concurrent operations\n")
        f.write("✅ **Schema Evolution:** Add columns without breaking existing queries\n")
        f.write("✅ **Time Travel:** Query historical versions of data\n")
        f.write("✅ **No Partitioning:** Data stored without folder partitioning\n")
        f.write("✅ **Parquet Source:** Direct load from parquet files\n\n")

        f.write("---\n")
        f.write("*Generated with PyIceberg and Apache Iceberg*\n")

    logger.info("✓ Results saved to %s", output_file)


def main() -> None:
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("APACHE ICEBERG TABLE CREATION")
    logger.info("=" * 60)

    try:
        # Initialize manager
        manager = IcebergManager()

        # Create Iceberg table
        table_name = manager.create_iceberg_table()

        # Get record count
        table = manager.catalog.load_table(table_name)
        scan = table.scan()
        record_count = len(scan.to_arrow())

        # Demonstrate schema evolution
        schema_evolution_result = manager.add_simple_column(table_name)

        # Demonstrate time travel
        time_travel_result = manager.demonstrate_time_travel(table_name)

        # Compile results
        results = {
            "table_name": table_name,
            "record_count": record_count,
            "schema_evolution": schema_evolution_result,
            "time_travel": time_travel_result,
            "timestamp": datetime.now().isoformat()
        }

        # Save to markdown
        save_results_to_markdown(results)

        # Close connections
        manager.close()

        logger.info("🎉 Iceberg table creation completed successfully!")
        logger.info("📄 Check 'iceberg_features_demo.md' for detailed results")

    except Exception as e:
        logger.error("❌ Demo failed: %s", e)
        raise


if __name__ == "__main__":
    main()