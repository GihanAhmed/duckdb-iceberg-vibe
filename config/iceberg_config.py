"""
Apache Iceberg Configuration for Space Analytics Demo
"""
import os
from typing import Dict, Any


class IcebergConfig:
    """Configuration settings for Apache Iceberg."""

    def __init__(self):
        self.warehouse_path = os.getenv("ICEBERG_WAREHOUSE_PATH", "data/iceberg_warehouse")
        self.catalog_type = "local"  # or "s3" for cloud storage
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        
    def get_catalog_properties(self) -> Dict[str, Any]:
        """Get Iceberg catalog properties."""
        if self.catalog_type == "local":
            return {
                "type": "rest",
                "uri": f"file://{os.path.abspath(self.warehouse_path)}",
                "warehouse": f"file://{os.path.abspath(self.warehouse_path)}"
            }
        elif self.catalog_type == "s3":
            if not all([self.aws_access_key, self.aws_secret_key]):
                raise ValueError("AWS credentials required for S3 catalog")
            
            return {
                "type": "rest", 
                "uri": f"s3://{self.warehouse_path}",
                "warehouse": f"s3://{self.warehouse_path}",
                "s3.access-key-id": self.aws_access_key,
                "s3.secret-access-key": self.aws_secret_key,
                "s3.region": self.aws_region
            }
        else:
            raise ValueError(f"Unsupported catalog type: {self.catalog_type}")
    
    def get_table_properties(self) -> Dict[str, str]:
        """Get default table properties for Iceberg tables."""
        return {
            "format-version": "2",
            "write.target-file-size-bytes": "134217728",  # 128MB
            "write.delete.target-file-size-bytes": "67108864",  # 64MB
            "write.parquet.compression-codec": "snappy",
            "write.metadata.compression-codec": "gzip"
        }
    
    def get_partition_spec(self) -> list[str]:
        """Get partition specification for NEO data."""
        return ["approach_year"]  # Partition by year for time-based queries


# Default configuration instance
iceberg_config = IcebergConfig()