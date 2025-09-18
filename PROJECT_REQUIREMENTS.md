# Claude Code PRD: Space Analytics DuckDB + Iceberg Demo

## Project Context
**Project Name:** Cosmic Analytics - DuckDB + Iceberg Showcase  
**Developer:** Data Engineer preparing presentation demo  
**Timeline:** 6 days development + presentation  
**Primary Goal:** Create compelling technical demonstration of modern lakehouse architecture using space data

## Technical Stack Requirements

### Core Technologies
- **Database:** DuckDB (local) + MotherDuck (cloud)
- **Storage Format:** Apache Iceberg tables
- **Language:** Python 3.11+ with SQL
- **Development Environment:** VSCode + Claude Code
- **Cloud Storage:** AWS S3 (optional)
- **Visualization:** matplotlib, plotly, seaborn

### Required Python Packages
```txt
duckdb>=0.9.0
pandas>=2.0.0
pyiceberg>=0.5.0
boto3>=1.26.0
matplotlib>=3.6.0
plotly>=5.15.0
seaborn>=0.12.0
jupyter>=1.0.0
requests>=2.28.0
pyarrow>=12.0.0
```

## Project Structure Requirements
```
space-analytics-demo/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── iceberg_warehouse/
├── scripts/
│   ├── 01_setup_environment.py
│   ├── 02_data_ingestion.py
│   ├── 03_duckdb_setup.py
│   ├── 04_iceberg_conversion.py
│   ├── 05_performance_benchmarks.py --
│   ├── 06_advanced_analytics.py --
│   └── 07_visualization.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── performance_analysis.ipynb
│   └── presentation_demo.ipynb
├── config/
│   ├── duckdb_config.py
│   └── iceberg_config.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── performance_tester.py
│   ├── risk_calculator.py
│   └── visualizer.py
└── tests/
    ├── test_data_loader.py
    ├── test_performance.py
    └── test_analytics.py
```

## Data Requirements & Specifications

### Primary Dataset: NASA NEO Close Approaches
- **Source URL:** `https://cneos.jpl.nasa.gov/ca/`
- **Format:** CSV (~10MB, 50K records)
- **Key Columns:**
  ```
  des: Object designation (VARCHAR)
  orbit_id: Orbit solution ID (VARCHAR) 
  jd: Julian date (DOUBLE)
  cd: Calendar date (VARCHAR)
  dist: Nominal approach distance (AU) (DOUBLE)
  dist_min: Minimum approach distance (AU) (DOUBLE)
  dist_max: Maximum approach distance (AU) (DOUBLE)
  v_rel: Velocity relative to Earth (km/s) (DOUBLE)
  v_inf: Velocity relative to infinity (km/s) (DOUBLE)
  t_sigma_f: 3-sigma uncertainty (VARCHAR)
  h: Absolute magnitude (DOUBLE)
  fullname: Object full name (VARCHAR)
  ```

### Data Quality Requirements
- Handle NULL values in h (magnitude) column
- Parse dates correctly (cd column format: YYYY-MMM-DD HH:MM)
- Validate distance values (dist > 0)
- Clean object names and designations

## Code Implementation Requirements

### 1. Environment Setup Script
**File:** `scripts/01_setup_environment.py`
**Requirements:**
```python
# Must include:
# - Virtual environment validation
# - Package installation verification
# - Directory structure creation
# - Configuration file setup
# - MotherDuck connection test
# - AWS credentials validation (if using S3)

def verify_environment():
    """Verify all required packages and connections"""
    pass

def create_directory_structure():
    """Create required project directories"""
    pass

def test_connections():
    """Test DuckDB, MotherDuck, and S3 connections"""
    pass
```

### 2. Data Ingestion Pipeline
**File:** `scripts/02_data_ingestion.py`
**Requirements:**
```python
import pandas as pd
import requests
import duckdb
from pathlib import Path

class NEODataIngester:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.base_url = "https://data.nasa.gov/resource/2rkh-ecvw.csv"
    
    def download_neo_data(self, limit: int = 50000) -> pd.DataFrame:
        """Download NEO close approach data from NASA API"""
        # Implementation required:
        # - Error handling for network requests
        # - Data validation
        # - Progress indicators
        # - Retry logic
        pass
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate NEO data"""
        # Implementation required:
        # - Date parsing and validation
        # - NULL value handling
        # - Data type conversions
        # - Outlier detection
        pass
    
    def save_data(self, df: pd.DataFrame, format: str = "csv") -> str:
        """Save data in specified format"""
        # Support: CSV, Parquet, JSON
        pass

# Usage example that must work:
ingester = NEODataIngester()
raw_data = ingester.download_neo_data()
clean_data = ingester.clean_data(raw_data)
file_path = ingester.save_data(clean_data, "parquet")
```

### 3. DuckDB Setup & Operations
**File:** `scripts/03_duckdb_setup.py`
**Requirements:**
```python
import duckdb
from typing import Dict, List, Any

class DuckDBManager:
    def __init__(self, db_path: str = "space_analytics.db"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
    
    def create_neo_table(self, data_path: str) -> None:
        """Create NEO table from data file"""
        # Required SQL operations:
        # - DROP TABLE IF EXISTS
        # - CREATE TABLE with proper schema
        # - Data type optimization
        # - Index creation for performance
        pass
    
    def setup_motherduck_sync(self, token: str) -> None:
        """Setup MotherDuck connection and sync"""
        # Implementation required:
        # - Authentication
        # - Database creation on MotherDuck
        # - Data synchronization
        # - Connection pooling
        pass
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute query and return results"""
        pass
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive table statistics"""
        # Return: row count, column info, storage size, etc.
        pass

# Required sample queries that must execute:
SAMPLE_QUERIES = {
    "total_count": "SELECT COUNT(*) FROM neo_approaches",
    "dangerous_objects": """
        SELECT des, fullname, h, dist, v_rel 
        FROM neo_approaches 
        WHERE h < 18 AND dist < 0.05 
        ORDER BY dist ASC
    """,
    "yearly_discoveries": """
        SELECT 
            YEAR(CAST(cd AS DATE)) as year,
            COUNT(*) as approaches,
            AVG(dist) as avg_distance,
            MIN(dist) as closest_approach
        FROM neo_approaches 
        WHERE cd IS NOT NULL
        GROUP BY year 
        ORDER BY year
    """
}
```

### 4. Iceberg Implementation
**File:** `scripts/04_iceberg_conversion.py`
**Requirements:**
```python
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import *
import pyarrow as pa

class IcebergManager:
    def __init__(self, warehouse_path: str = "data/iceberg_warehouse"):
        self.warehouse_path = warehouse_path
        self.catalog = self._setup_catalog()
    
    def _setup_catalog(self):
        """Setup Iceberg catalog (local or S3)"""
        # Support both local and S3 catalogs
        pass
    
    def create_neo_iceberg_table(self, data_source: str) -> str:
        """Convert NEO data to Iceberg format"""
        # Required schema definition:
        schema = Schema(
            NestedField(1, "des", StringType()),
            NestedField(2, "orbit_id", StringType()),
            NestedField(3, "jd", DoubleType()),
            NestedField(4, "cd", StringType()),
            NestedField(5, "dist", DoubleType()),
            NestedField(6, "dist_min", DoubleType()),
            NestedField(7, "dist_max", DoubleType()),
            NestedField(8, "v_rel", DoubleType()),
            NestedField(9, "v_inf", DoubleType()),
            NestedField(10, "t_sigma_f", StringType()),
            NestedField(11, "h", DoubleType()),
            NestedField(12, "fullname", StringType()),
            NestedField(13, "approach_year", IntegerType()),  # Partition column
        )
        
        # Implementation requirements:
        # - Table creation with partitioning
        # - Data loading from DuckDB
        # - Metadata optimization
        pass
    
    def demonstrate_time_travel(self, table_name: str) -> Dict[str, Any]:
        """Show Iceberg time travel capabilities"""
        # Required demonstrations:
        # - Historical snapshots
        # - Table history
        # - Rollback operations
        # - Schema evolution
        pass
    
    def perform_table_maintenance(self, table_name: str) -> None:
        """Perform Iceberg table maintenance"""
        # Operations: expire snapshots, compact files, etc.
        pass

# Required integration with DuckDB
def setup_duckdb_iceberg_integration(duckdb_conn, iceberg_table_path: str):
    """Setup DuckDB to query Iceberg tables"""
    # Must support querying Iceberg tables from DuckDB
    pass
```

### 5. Performance Benchmarking
**File:** `scripts/05_performance_benchmarks.py`
**Requirements:**
```python
import time
import statistics
from typing import Dict, List, Callable
import matplotlib.pyplot as plt

class PerformanceBenchmarker:
    def __init__(self):
        self.results = {}
    
    def benchmark_query(self, query: str, connection, iterations: int = 5) -> Dict[str, float]:
        """Benchmark a single query multiple times"""
        times = []
        for _ in range(iterations):
            start = time.time()
            connection.execute(query).fetchall()
            end = time.time()
            times.append(end - start)
        
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def compare_storage_formats(self, base_query: str) -> Dict[str, Dict[str, float]]:
        """Compare query performance across storage formats"""
        # Test against: CSV, Parquet, Iceberg
        # Required queries:
        formats = {
            "csv": "SELECT * FROM read_csv_auto('data/neo_approaches.csv')",
            "parquet": "SELECT * FROM 'data/neo_approaches.parquet'",
            "iceberg": "SELECT * FROM iceberg.space_analytics.neo_approaches"
        }
        # Implementation required
        pass
    
    def storage_efficiency_analysis(self) -> Dict[str, Any]:
        """Analyze storage efficiency of different formats"""
        # Compare: file sizes, compression ratios, query performance
        pass
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        # Create markdown report with charts
        pass

# Required benchmark queries (must all execute successfully):
BENCHMARK_QUERIES = {
    "simple_count": "SELECT COUNT(*) FROM neo_approaches",
    "filtered_count": "SELECT COUNT(*) FROM neo_approaches WHERE h < 20",
    "aggregation": "SELECT approach_year, COUNT(*), AVG(dist) FROM neo_approaches GROUP BY approach_year",
    "complex_join": """
        SELECT a.des, COUNT(a.des) as approaches, MIN(a.dist) as closest
        FROM neo_approaches a 
        JOIN neo_approaches b ON a.des = b.des 
        WHERE a.h IS NOT NULL 
        GROUP BY a.des 
        HAVING COUNT(a.des) > 1
    """,
    "analytical": """
        SELECT 
            des,
            h,
            dist,
            v_rel,
            CASE 
                WHEN h < 18 AND dist < 0.05 THEN 'HIGH_RISK'
                WHEN h < 22 AND dist < 0.1 THEN 'MEDIUM_RISK'
                ELSE 'LOW_RISK'
            END as risk_category,
            ROW_NUMBER() OVER (ORDER BY dist ASC) as risk_rank
        FROM neo_approaches 
        WHERE h IS NOT NULL AND dist IS NOT NULL
        ORDER BY dist ASC
        LIMIT 100
    """
}
```

### 6. Advanced Analytics Implementation
**File:** `scripts/06_advanced_analytics.py`
**Requirements:**
```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import duckdb

class SimpleAnalytics:
    def __init__(self, duckdb_conn):
        self.conn = duckdb_conn
        
    def calculate_risk_score(self, h: float, dist: float, v_rel: float) -> float:
        """Simple risk calculation: bigger, closer, faster = more risk"""
        if any(pd.isna(x) for x in [h, dist, v_rel]):
            return 0.0
        
        # Simple scoring (0-100):
        size_score = max(0, 30 - h) * 2  # H=30 is small, H=10 is huge
        distance_score = max(0, 0.1 - dist) * 500  # < 0.1 AU is very close
        speed_score = min(v_rel, 50) * 2  # Up to 50 km/s
        
        return min(size_score + distance_score + speed_score, 100)
    
    def find_risky_objects(self, top_n: int = 20) -> pd.DataFrame:
        """Find the most risky near-Earth objects"""
        # Simple query with risk calculation
        pass
    
    def analyze_trends_by_year(self) -> pd.DataFrame:
        """Analyze discovery trends over time"""
        # Simple yearly aggregation
        pass

# Simple analytics queries:
ANALYTICS_QUERIES = {
    "top_risks": """
        SELECT des, fullname, h, dist, v_rel,
               (GREATEST(0, 30 - h) * 2 + 
                GREATEST(0, 0.1 - dist) * 500 + 
                LEAST(v_rel, 50) * 2) as risk_score
        FROM neo_data 
        WHERE h IS NOT NULL AND dist IS NOT NULL AND v_rel IS NOT NULL
        ORDER BY risk_score DESC LIMIT 20
    """,
    
    "yearly_trends": """
        SELECT YEAR(CAST(cd AS DATE)) as year,
               COUNT(*) as total_approaches,
               COUNT(DISTINCT des) as unique_objects,
               AVG(dist) as avg_distance,
               MIN(dist) as closest_approach
        FROM neo_data
        WHERE cd IS NOT NULL
        GROUP BY year ORDER BY year
    """,
    
    "size_categories": """
        SELECT 
            CASE 
                WHEN h < 18 THEN 'Very Large (>1km)'
                WHEN h < 22 THEN 'Large (140m-1km)' 
                WHEN h < 25 THEN 'Medium (30-140m)'
                ELSE 'Small (<30m)'
            END as size_category,
            COUNT(*) as count,
            MIN(dist) as closest_ever
        FROM neo_data
        WHERE h IS NOT NULL
        GROUP BY size_category
        ORDER BY MIN(h)
    """
}
```

### 7. Visualization & Presentation Tools
**File:** `scripts/07_visualization.py`
**Requirements:**
```python
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd

class SpaceDataVisualizer:
    def __init__(self, data_source):
        self.data_source = data_source
        self.setup_style()
    
    def setup_style(self):
        """Setup consistent styling for all charts"""
        plt.style.use('dark_background')  # Space theme
        sns.set_palette("viridis")
    
    def create_performance_comparison_chart(self, benchmark_results: Dict) -> go.Figure:
        """Create interactive performance comparison chart"""
        # Required chart types:
        # - Bar chart comparing query times
        # - Box plots showing performance distribution
        # - Line chart for different data sizes
        # Must be interactive with plotly
        pass
    
    def create_risk_assessment_dashboard(self, risk_data: pd.DataFrame) -> go.Figure:
        """Create comprehensive risk assessment dashboard"""
        # Required subplots:
        # 1. Scatter plot: Size vs Distance (colored by risk)
        # 2. Histogram: Risk score distribution
        # 3. Time series: Risk trends over time
        # 4. Bar chart: Top 10 riskiest objects
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk vs Size & Distance', 'Risk Distribution', 
                          'Risk Trends Over Time', 'Highest Risk Objects'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        # Implementation required
        pass
    
    def create_discovery_timeline(self, discovery_data: pd.DataFrame) -> go.Figure:
        """Create interactive discovery timeline"""
        # Animated timeline showing discoveries over time
        # Must include size and risk indicators
        pass
    
    def create_storage_efficiency_comparison(self, efficiency_data: Dict) -> go.Figure:
        """Create storage format comparison visualization"""
        # Compare CSV, Parquet, and Iceberg formats
        # Metrics: file size, query speed, compression ratio
        pass
    
    def generate_presentation_slides(self) -> List[go.Figure]:
        """Generate all presentation-ready visualizations"""
        # Return list of publication-ready figures
        pass

# Required chart configurations:
CHART_CONFIGS = {
    "performance_chart": {
        "title": "DuckDB + Iceberg vs Traditional Storage",
        "x_label": "Storage Format",
        "y_label": "Query Time (seconds)",
        "color_scheme": "plasma"
    },
    "risk_dashboard": {
        "title": "Near-Earth Object Risk Assessment",
        "size_range": [10, 100],
        "color_scale": "Reds"
    },
    "timeline": {
        "title": "NEO Discovery Timeline",
        "animation_frame": "year",
        "size_max": 50
    }
}

# Required export formats:
EXPORT_FORMATS = ["png", "pdf", "html", "svg"]
```

## Testing Requirements

### Unit Tests Required
**File:** `tests/test_data_loader.py`
```python
def test_neo_data_download():
    """Test NEO data download functionality"""
    pass

def test_data_cleaning():
    """Test data cleaning and validation"""
    pass

def test_data_type_conversion():
    """Test proper data type handling"""
    pass
```

**File:** `tests/test_performance.py`
```python
def test_benchmark_execution():
    """Test all benchmark queries execute successfully"""
    pass

def test_format_comparison():
    """Test storage format performance comparison"""
    pass
```

## Demo Script Requirements

### Presentation Notebook
**File:** `notebooks/presentation_demo.ipynb`
**Must include executable cells for:**

1. **Environment verification** (30 seconds)
2. **Data loading demonstration** (1 minute)  
3. **DuckDB query showcase** (2 minutes)
4. **Iceberg conversion** (2 minutes)
5. **Performance comparison** (3 minutes)
6. **Advanced analytics** (4 minutes)
7. **Time travel demonstration** (2 minutes)
8. **Visualization showcase** (1 minute)

### Live Demo Commands
```python
# Must work flawlessly during presentation:

# 1. Quick data exploration
conn = duckdb.connect('space_analytics.db')
print(conn.execute("SELECT COUNT(*) FROM neo_approaches").fetchone())

# 2. Risk assessment query
risk_query = """
SELECT des, fullname, calculate_neo_risk(h, dist, v_rel) as risk 
FROM neo_approaches 
ORDER BY risk DESC LIMIT 5
"""
print(conn.execute(risk_query).fetchdf())

# 3. Performance comparison
formats = ['csv', 'parquet', 'iceberg']
for fmt in formats:
    start = time.time()
    conn.execute(f"SELECT COUNT(*) FROM neo_{fmt}").fetchone()
    print(f"{fmt}: {time.time() - start:.3f}s")

# 4. Iceberg time travel
conn.execute("""
SELECT * FROM iceberg.space_analytics.neo_approaches 
FOR SYSTEM_TIME AS OF '2024-01-01 00:00:00'
LIMIT 5
""").fetchdf()
```

## Success Criteria Checklist

### Code Quality Requirements ✅
- [ ] All scripts execute without errors
- [ ] Comprehensive error handling implemented
- [ ] Code is well-documented with docstrings
- [ ] Type hints used throughout
- [ ] Consistent code formatting (black/autopep8)

### Performance Requirements ✅
- [ ] Iceberg queries ≥2x faster than CSV
- [ ] All benchmark queries complete <5 seconds
- [ ] Memory usage <4GB during demo
- [ ] Storage savings ≥50% vs CSV

### Demonstration Requirements ✅
- [ ] All demo queries execute flawlessly
- [ ] Visualizations render correctly
- [ ] Time travel functionality works
- [ ] Risk calculations are accurate
- [ ] Performance comparisons show clear benefits

### Documentation Requirements ✅
- [ ] Complete README with setup instructions
- [ ] Inline code comments explaining logic
- [ ] Query explanations for non-technical audience
- [ ] Troubleshooting guide for common issues

## Code Guidelines & Standards

### Code Quality Requirements

#### Style & Formatting
- **Python Style:** Follow PEP 8 standards strictly
- **Line Length:** Maximum 88 characters (Black formatter compatible)
- **Imports:** Group imports (standard library, third-party, local) with blank lines
- **Naming Conventions:**
  - Classes: `PascalCase` (e.g., `SimpleDataLoader`)
  - Functions/methods: `snake_case` (e.g., `download_nasa_data`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `API_BASE_URL`)
  - Variables: `snake_case` (e.g., `neo_data`, `query_result`)

#### Documentation Standards
- **Docstrings:** All classes and public methods must have docstrings
- **Type Hints:** Use type hints for all function parameters and return types
- **Comments:** Explain complex logic, not obvious code
- **README:** Each script must have a header comment explaining purpose

```python
# Example of required documentation style:
from typing import Dict, List, Optional
import pandas as pd

class SimpleDataLoader:
    """
    Handles downloading and processing NASA JPL Close Approach Data.
    
    This class provides a simple interface to fetch Near-Earth Object data
    from NASA's JPL API and convert it to various formats for analysis.
    
    Attributes:
        api_url (str): Base URL for NASA JPL CAD API
        params (Dict[str, str]): Default query parameters
    """
    
    def __init__(self, date_range: tuple = ("1900-01-01", "2030-12-31")) -> None:
        """Initialize the data loader with date range parameters."""
        self.api_url = "https://ssd-api.jpl.nasa.gov/cad.api"
        self.params = self._build_params(date_range)
    
    def download_data(self, max_retries: int = 3) -> Dict[str, any]:
        """
        Download NEO data from NASA JPL API with retry logic.
        
        Args:
            max_retries: Maximum number of retry attempts for failed requests
            
        Returns:
            Dictionary containing API response with 'data' and 'fields' keys
            
        Raises:
            requests.RequestException: If API request fails after all retries
            ValueError: If API returns invalid or empty data
        """
        # Implementation here...
        pass
```

#### Error Handling Standards
- **Always use try-except blocks** for external API calls, file operations, and database connections
- **Specific exception types:** Catch specific exceptions, not generic `Exception`
- **Logging:** Use proper logging instead of print statements
- **Graceful degradation:** Provide fallback options when possible

```python
import logging
import requests
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_with_fallback(url: str, backup_file: Optional[str] = None) -> Dict:
    """Example of proper error handling with fallback."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        logger.warning("API request timed out, attempting backup...")
        if backup_file and os.path.exists(backup_file):
            return self._load_backup_data(backup_file)
        raise
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        if backup_file:
            logger.info(f"Loading backup data from {backup_file}")
            return self._load_backup_data(backup_file)
        raise
    
    except ValueError as e:
        logger.error(f"Invalid API response format: {e}")
        raise
```

#### Testing Requirements
- **Unit Tests:** Every public method must have unit tests
- **Integration Tests:** Test complete workflows end-to-end
- **Mock External Dependencies:** Use mocks for API calls and file operations
- **Test Data:** Include sample datasets for testing
- **Coverage:** Aim for >80% code coverage

```python
# Example test structure required:
import pytest
from unittest.mock import Mock, patch
import pandas as pd

class TestSimpleDataLoader:
    """Test suite for SimpleDataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = SimpleDataLoader()
        self.sample_api_response = {
            "signature": {"version": "1.5"},
            "count": 2,
            "fields": ["des", "cd", "dist", "h"],
            "data": [
                ["433", "2024-Apr-13 12:00", "0.002", "17.7"],
                ["99942", "2029-Apr-13 21:46", "0.0002", "19.7"]
            ]
        }
    
    @patch('requests.get')
    def test_download_data_success(self, mock_get):
        """Test successful data download from API."""
        mock_response = Mock()
        mock_response.json.return_value = self.sample_api_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.loader.download_data()
        
        assert result["count"] == 2
        assert len(result["data"]) == 2
        mock_get.assert_called_once()
    
    def test_data_validation(self):
        """Test data validation and cleaning."""
        # Test with invalid data scenarios
        pass
```

#### Performance Standards
- **SQL Optimization:** All demo queries must execute in <2 seconds
- **Memory Management:** Handle large datasets without excessive memory usage
- **Lazy Loading:** Load data on-demand when possible
- **Connection Pooling:** Reuse database connections
- **Caching:** Cache API responses and expensive computations

```python
# Example of performance-conscious code:
import functools
from typing import Dict, Any

class PerformanceOptimizedLoader:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._connection = None
    
    @functools.lru_cache(maxsize=128)
    def get_cached_query_result(self, query_hash: str) -> pd.DataFrame:
        """Cache expensive query results."""
        pass
    
    @property
    def db_connection(self):
        """Lazy-loaded database connection with connection pooling."""
        if self._connection is None:
            self._connection = duckdb.connect("space.db")
        return self._connection
    
    def __del__(self):
        """Cleanup connections on object destruction."""
        if self._connection:
            self._connection.close()
```

#### Security Guidelines
- **API Keys:** Store sensitive data in environment variables, never in code
- **Input Validation:** Validate all user inputs and API parameters
- **SQL Injection Prevention:** Use parameterized queries
- **File Path Security:** Validate file paths to prevent directory traversal

```python
import os
from pathlib import Path

class SecureDataHandler:
    def __init__(self):
        # Secure environment variable handling
        self.motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
        if not self.motherduck_token:
            raise ValueError("MOTHERDUCK_TOKEN environment variable required")
        
        # Secure file path handling
        self.data_dir = Path("data").resolve()
        self.data_dir.mkdir(exist_ok=True)
    
    def save_secure_file(self, filename: str, data: Any) -> Path:
        """Securely save files within project directory."""
        # Prevent directory traversal
        safe_path = self.data_dir / Path(filename).name
        if not safe_path.is_relative_to(self.data_dir):
            raise ValueError(f"Invalid file path: {filename}")
        return safe_path
```

### Development Workflow Standards

#### Git Workflow
- **Commits:** Small, focused commits with descriptive messages
- **Branches:** Feature branches for each major component
- **Commit Messages:** Follow conventional commit format:
  ```
  feat: add NASA API data loader with retry logic
  fix: resolve DuckDB connection timeout issues
  docs: update README with setup instructions
  test: add unit tests for risk calculation
  ```

#### Code Review Checklist
Before considering any code complete, verify:
- [ ] All functions have type hints and docstrings
- [ ] Error handling covers expected failure modes
- [ ] Unit tests written and passing
- [ ] No hardcoded secrets or API keys
- [ ] Performance requirements met
- [ ] Code follows PEP 8 standards
- [ ] Dependencies properly managed in requirements.txt

#### Demo-Specific Requirements
- **Reliability:** All demo code must work flawlessly during presentation
- **Timing:** Each demo segment must complete within allocated time
- **Fallbacks:** Backup plans for every live demo component
- **Clarity:** Code must be readable when projected on screen

```python
# Demo code must be extra clean and well-commented:
class DemoAnalytics:
    """
    DEMO CLASS: Simple analytics for live presentation.
    
    This class is optimized for presentation clarity and reliability.
    All methods are designed to execute quickly and handle errors gracefully.
    """
    
    def show_most_dangerous_asteroids(self, limit: int = 10) -> pd.DataFrame:
        """
        DEMO METHOD: Find and display the most dangerous Near-Earth Objects.
        
        This method will be executed live during the presentation.
        Results are formatted for easy reading on screen.
        """
        try:
            # Simple, reliable query that always works
            query = """
            SELECT des as "Object", 
                   fullname as "Name",
                   ROUND(dist, 6) as "Distance (AU)",
                   ROUND(h, 1) as "Size (H mag)",
                   ROUND(v_rel, 1) as "Velocity (km/s)"
            FROM neo_data 
            WHERE h IS NOT NULL AND dist IS NOT NULL 
            ORDER BY (30 - h) + (0.1 - dist) * 100 DESC
            LIMIT ?
            """
            
            result = self.conn.execute(query, [limit]).df()
            
            # Format for presentation display
            result.columns = ["🛸 Object", "📛 Name", "📏 Distance", "📐 Size", "⚡ Speed"]
            
            return result
            
        except Exception as e:
            # Graceful fallback for demo
            logger.warning(f"Demo query failed: {e}")
            return self._get_demo_fallback_data()
```

#### Validation & Testing Workflow
1. **Unit Test First:** Write tests before implementing functionality
2. **Integration Testing:** Test complete workflows with real data
3. **Performance Validation:** Benchmark all demo queries
4. **Demo Rehearsal:** Practice complete presentation flow
5. **Backup Testing:** Verify all fallback scenarios work

#### Required Validation Commands
All these commands must succeed before demo:
```bash
# Code quality checks
black --check scripts/
flake8 scripts/
mypy scripts/

# Testing
python -m pytest tests/ -v --cov=scripts --cov-report=html
python scripts/validate_demo.py

# Performance validation  
python scripts/benchmark_all_queries.py

# Demo rehearsal
jupyter nbconvert --execute notebooks/demo_presentation.ipynb

# End-to-end test
python scripts/full_demo_test.py
```

### Claude Code Specific Instructions

#### Development Approach
1. **Code Generation:** Always generate complete, working implementations
2. **Error Handling:** Include comprehensive error handling from the start
3. **Testing:** Generate tests alongside implementation code
4. **Documentation:** Create docstrings and comments as code is written
5. **Validation:** Test all code before considering it complete

#### Code Review Focus Areas
- **Functionality:** Does the code work as specified?
- **Performance:** Does it meet demo timing requirements?
- **Reliability:** Will it work during live presentation?
- **Readability:** Can the audience understand it when projected?
- **Maintainability:** Is it well-structured and documented?

### Required Validations Before Demo
```bash
# All these commands must succeed:
python scripts/01_setup_environment.py
python scripts/02_data_ingestion.py
python scripts/03_duckdb_setup.py
python scripts/04_iceberg_conversion.py  
python scripts/05_performance_benchmarks.py
python scripts/06_advanced_analytics.py
python scripts/07_visualization.py
python -m pytest tests/ -v
jupyter nbconvert --execute notebooks/presentation_demo.ipynb
```

## Implementation Phases for Claude Code

### Phase 1: Foundation Setup (Day 1)
**Claude Code Commands:**
```bash
# Start here - Claude Code will read PRD and create structure
"Read the PROJECT_REQUIREMENTS.md file and create the complete project directory structure"

"Implement scripts/01_setup_environment.py with all environment validations from the PRD"

"Create the requirements.txt file with all specified package versions"

"Build the config/duckdb_config.py and config/iceberg_config.py configuration files"

"Generate a comprehensive README.md with setup instructions"
```

**Expected Outputs:**
- Complete directory structure created
- Environment setup script that validates all dependencies
- Configuration files for DuckDB and Iceberg
- README with clear setup instructions
- Working virtual environment with all packages

### Phase 2: Data Pipeline Development (Day 1-2)
**Claude Code Commands:**
```bash
"Implement the NEODataIngester class in scripts/02_data_ingestion.py exactly as specified in the PRD"

"Create the DuckDBManager class in scripts/03_duckdb_setup.py with all required methods"

"Build unit tests for the data ingestion pipeline in tests/test_data_loader.py"

"Generate sample data download and cleaning scripts that handle all edge cases"
```

**Expected Outputs:**
- Working data ingestion pipeline
- DuckDB database creation and management
- Comprehensive error handling
- Unit tests passing
- Sample data successfully loaded

### Phase 3: Iceberg Integration (Day 2-3)
**Claude Code Commands:**
```bash
"Implement the IcebergManager class in scripts/04_iceberg_conversion.py with full schema definition"

"Create DuckDB-Iceberg integration functions that enable querying Iceberg tables from DuckDB"

"Build time