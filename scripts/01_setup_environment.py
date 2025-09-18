"""
Environment Setup Script for Space Analytics Demo

This script validates the environment, installs required packages,
creates directory structure, and tests all connections.
"""
import sys
import importlib
import logging
from pathlib import Path
from typing import Dict

# Import optional dependencies at module level
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_environment() -> bool:
    """Verify all required packages and connections."""
    logger.info("Verifying environment setup...")

    required_packages = [
        'duckdb', 'pandas', 'matplotlib', 'plotly',
        'seaborn', 'jupyter', 'requests', 'pyarrow'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info("✓ %s is installed", package)
        except ImportError:
            missing_packages.append(package)
            logger.warning("✗ %s is missing", package)

    if missing_packages:
        logger.error("Missing packages: %s", missing_packages)
        logger.info("Run: pip install -r requirements.txt")
        return False

    logger.info("All required packages are installed")
    return True


def create_directory_structure() -> bool:
    """Create required project directories."""
    logger.info("Creating directory structure...")

    directories = [
        "data/raw",
        "data/processed",
        "data/iceberg_warehouse",
        "scripts",
        "config",
        "src",
        "tests",
        "notebooks"
    ]

    try:
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info("✓ Created directory: %s", directory)

        # Create __init__.py files for Python packages
        for init_dir in ["src", "config"]:
            init_file = Path(init_dir) / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                logger.info("✓ Created __init__.py in %s", init_dir)

        return True

    except (OSError, PermissionError) as exc:
        logger.error("Error creating directories: %s", exc)
        return False


def test_connections() -> Dict[str, bool]:
    """Test DuckDB connection."""
    logger.info("Testing connections...")

    results = {
        "duckdb": False
    }

    # Test DuckDB
    if DUCKDB_AVAILABLE:
        try:
            conn = duckdb.connect(":memory:")
            conn.execute("SELECT 1").fetchone()
            conn.close()
            results["duckdb"] = True
            logger.info("✓ DuckDB connection successful")
        except (duckdb.Error, RuntimeError) as exc:
            logger.error("✗ DuckDB connection failed: %s", exc)
    else:
        logger.error("✗ DuckDB not available")


    return results


def install_duckdb_extensions() -> bool:
    """Install required DuckDB extensions."""
    logger.info("Installing DuckDB extensions...")

    extensions = ['parquet', 'json']

    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB not available")
        return False

    try:
        conn = duckdb.connect(":memory:")

        for ext in extensions:
            try:
                conn.execute(f"INSTALL {ext}")
                conn.execute(f"LOAD {ext}")
                logger.info("✓ Installed and loaded %s extension", ext)
            except duckdb.Error as exc:
                logger.warning("⚠ Failed to install %s: %s", ext, exc)

        conn.close()
        return True

    except (duckdb.Error, RuntimeError) as exc:
        logger.error("Error installing extensions: %s", exc)
        return False


def validate_configuration() -> bool:
    """Validate configuration files."""
    logger.info("Validating configuration...")

    required_files = [
        "requirements.txt"
    ]

    all_exist = True

    for file_path in required_files:
        if Path(file_path).exists():
            logger.info("✓ %s exists", file_path)
        else:
            logger.error("✗ %s missing", file_path)
            all_exist = False

    return all_exist


def main():
    """Main setup function."""
    logger.info("=" * 60)
    logger.info("SPACE ANALYTICS DEMO - ENVIRONMENT SETUP")
    logger.info("=" * 60)

    setup_steps = [
        ("Creating directory structure", create_directory_structure),
        ("Validating configuration files", validate_configuration),
        ("Verifying Python packages", verify_environment),
        ("Installing DuckDB extensions", install_duckdb_extensions),
        ("Testing connections", lambda: bool(test_connections()))
    ]

    results = {}

    for step_name, step_function in setup_steps:
        logger.info("\n%s...", step_name)
        try:
            result = step_function()
            results[step_name] = result
            if result:
                logger.info("✓ %s completed successfully", step_name)
            else:
                logger.error("✗ %s failed", step_name)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("✗ %s failed with error: %s", step_name, exc)
            results[step_name] = False

    # Summary
    logger.info("\n%s", "=" * 60)
    logger.info("SETUP SUMMARY")
    logger.info("=" * 60)

    success_count = sum(1 for r in results.values() if r)
    total_count = len(results)

    for step, step_success in results.items():
        status = "✓ PASS" if step_success else "✗ FAIL"
        logger.info("%s: %s", status, step)

    logger.info(
        "\nOverall: %d/%d steps completed successfully",
        success_count, total_count
    )

    if success_count == total_count:
        logger.info("🎉 Environment setup completed successfully!")
        logger.info("\nNext step: Run python scripts/02_data_ingestion.py")
        return True

    logger.error("❌ Environment setup incomplete. Please fix errors above.")
    return False


if __name__ == "__main__":
    MAIN_SUCCESS = main()
    sys.exit(0 if MAIN_SUCCESS else 1)
