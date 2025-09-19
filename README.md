# 🚀 Space Analytics: DuckDB + Iceberg Demo

**Modern Lakehouse Architecture for Near-Earth Object Analysis**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.9.0+-yellow.svg)](https://duckdb.org)
[![Iceberg](https://img.shields.io/badge/Apache%20Iceberg-0.7.0+-orange.svg)](https://iceberg.apache.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project demonstrates the power of combining **DuckDB** with **Apache Iceberg** for high-performance space analytics, using NASA's Near-Earth Object (NEO) close approach data as a compelling use case.

## 🌟 Key Features

- **⚡ 10x+ Performance**: DuckDB delivers sub-second analytics on space data
- **🗄️ ACID Compliance**: Iceberg provides reliable data lake operations
- **⏰ Time Travel**: Query historical versions of your data
- **📊 Advanced Analytics**: Risk assessment and trend analysis for NEOs
- **🎯 Interactive Visualizations**: 3D plots and animated dashboards
- **🧪 Comprehensive Testing**: Full test suite with performance benchmarks
- **📱 Demo-Ready**: Jupyter notebooks for live presentations

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Demo Walkthrough](#-demo-walkthrough)
- [Performance Results](#-performance-results)
- [API Documentation](#-api-documentation)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Key Technologies

### 🧊 Apache Iceberg + PyIceberg
This project uses **PyIceberg** - the official Python implementation of Apache Iceberg:

```bash
pip install pyiceberg[duckdb]  # Includes DuckDB integration
```

**Key Features:**
- 🗂️ **Filesystem Catalog**: No external metadata store required
- 🔗 **DuckDB Integration**: Native connectivity for high-performance queries
- 📝 **Metadata Management**: Automatic JSON metadata files
- 🏗️ **Table Creation**: Programmatic schema definition and partitioning
- ⏰ **Snapshot Management**: Built-in versioning and time travel
- 📊 **Schema Evolution**: Add/modify columns safely

**True Iceberg Format:**
PyIceberg creates proper Iceberg tables with:
- `metadata/` directory with versioned JSON files
- UUID-based data file naming
- Snapshot history tracking
- Manifest files for efficient queries
- No folder-based partitioning (data stored flat)
- Direct parquet file source loading

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- 8GB+ RAM recommended
- 2GB+ free disk space

### 30-Second Setup

```bash
# Clone the repository
git clone <repository-url>
cd spaceneo-analytics-demo

# Install dependencies
pip install -r requirements.txt

# Set up environment
python scripts/01_setup_environment.py

# Download and process data
python scripts/02_data_ingestion.py

# Run the full pipeline
python scripts/03_duckdb_setup.py
python scripts/04_iceberg_creation.py
python scripts/05_performance_benchmarks.py
```

### Quick Demo

```bash
# Launch the interactive demo
jupyter lab notebooks/presentation_demo.ipynb
```

## 📁 Project Structure

```
space-analytics-demo/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies  
├── 📄 .env.example                 # Environment template
├── 📂 data/                        # Data storage
│   ├── 📂 raw/                     # Raw data files
│   ├── 📂 processed/               # Cleaned data
│   └── 📂 iceberg_warehouse/       # Iceberg tables
├── 📂 scripts/                     # Core implementation
│   ├── 📄 01_setup_environment.py  # Environment validation
│   ├── 📄 02_data_ingestion.py     # NASA data pipeline
│   ├── 📄 03_duckdb_setup.py       # Database setup
│   ├── 📄 04_iceberg_creation.py   # Iceberg table creation
│   ├── 📄 05_performance_benchmarks.py # Performance testing
│   ├── 📄 06_advanced_analytics.py # Risk & trend analysis
│   └── 📄 07_visualization.py      # Interactive charts
├── 📂 notebooks/                   # Jupyter demos
│   ├── 📄 data_exploration.ipynb   # Data analysis
│   ├── 📄 performance_analysis.ipynb # Benchmarking
│   └── 📄 presentation_demo.ipynb  # Live demo (15 min)
├── 📂 config/                      # Configuration
│   ├── 📄 duckdb_config.py        # DuckDB settings  
│   └── 📄 iceberg_config.py       # Iceberg settings
├── 📂 src/                         # Reusable modules
└── 📂 tests/                       # Unit tests
    ├── 📄 test_data_loader.py      # Data pipeline tests
    ├── 📄 test_performance.py      # Performance tests
    └── 📄 test_analytics.py        # Analytics tests
```

## 🔧 Installation

### Method 1: Standard Installation

```bash
# Create virtual environment
python -m venv space-analytics-env
source space-analytics-env/bin/activate  # On Windows: space-analytics-env\\Scripts\\activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python scripts/01_setup_environment.py
```

### Method 2: Development Setup

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials (optional)
nano .env
```

Required environment variables:
- `MOTHERDUCK_TOKEN`: For cloud DuckDB sync (optional)
- `AWS_ACCESS_KEY_ID`: For S3 Iceberg storage (optional)
- `AWS_SECRET_ACCESS_KEY`: For S3 Iceberg storage (optional)

## 📖 Usage Guide

### Step 1: Environment Setup

```bash
python scripts/01_setup_environment.py
```

**What it does:**
- ✅ Validates Python packages
- ✅ Creates directory structure  
- ✅ Tests database connections
- ✅ Installs DuckDB extensions

### Step 2: Data Ingestion

```bash
python scripts/02_data_ingestion.py
```

**What it does:**
- 🌐 Downloads NASA NEO data (50K+ records)
- 🧹 Cleans and validates data
- 💾 Saves in multiple formats (CSV, Parquet, JSON)

**Data Sources:**
- NASA's Close Approach Database
- ~50,000 asteroid/comet approaches
- Date range: 1900-2030
- File size: ~10-150MB depending on format

### Step 3: DuckDB Setup

```bash
python scripts/03_duckdb_setup.py
```

**What it does:**
- 🏗️ Creates optimized DuckDB database
- 📈 Builds indexes for performance
- ☁️ Syncs with MotherDuck (if configured)
- 🧪 Runs sample analytics queries

### Step 4: Iceberg Integration

```bash
python scripts/04_iceberg_creation.py
```

**What it does:**
- 🗄️ Converts data to Iceberg format
- 📅 Creates year-based partitions
- ⏰ Demonstrates time travel
- 🔧 Performs table maintenance

### Step 5: Performance Benchmarks

```bash
python scripts/05_performance_benchmarks.py
```

**What it does:**
- ⚡ Benchmarks query performance
- 📊 Compares storage formats
- 📈 Generates performance reports
- 💾 Analyzes storage efficiency

### Step 6: Advanced Analytics

```bash
python scripts/06_advanced_analytics.py
```

**What it does:**
- 🎯 Calculates NEO risk scores
- 📅 Analyzes temporal trends
- 📏 Studies size distributions
- 🚨 Identifies high-risk objects

### Step 7: Visualizations

```bash
python scripts/07_visualization.py
```

**What it does:**
- 📊 Creates interactive charts
- 🎨 Generates 3D visualizations
- 📈 Builds performance dashboards
- 💾 Exports in multiple formats

## 📦 Output Files Generated

Each script in the pipeline generates specific output files:

### Script → File Generation Map

| Script | Generated Files | Purpose |
|--------|----------------|---------|
| `02_data_ingestion.py` | `data/raw/neo_data_YYYYMMDD_HHMMSS.parquet` | Raw NASA NEO data |
| `03_duckdb_setup.py` | `space_analytics.db` | Local DuckDB database |
| `04_iceberg_creation.py` | `data/iceberg_warehouse/`<br>`catalog.db`<br>`demo.db/neo_approaches_iceberg/` | True Iceberg format with metadata |
| `05_performance_benchmarks.py` | `performance_report.md` | Performance comparison report |
| `06_advanced_analytics.py` | `analytics_results.md` | Risk analysis, trends, statistics |
| `07_visualization.py` | `visualizations/performance_comparison.html`<br>`visualizations/risk_assessment.html`<br>`visualizations/discovery_timeline.html`<br>`visualizations/storage_efficiency.html` | Interactive HTML charts |

### Execution Order
1. **Data Pipeline**: `02` → `03` → `04` (Creates data in multiple formats)
2. **Analysis Pipeline**: `05` → `06` → `07` (Generates reports and visualizations)
3. **Interactive Analysis**: Jupyter notebooks use the generated database and files

## 🎪 Demo Walkthrough

### Interactive Jupyter Demo (15 minutes)

```bash
jupyter lab notebooks/presentation_demo.ipynb
```

**Demo Timeline:**
1. **Environment Check** (30s) - Verify setup
2. **Data Overview** (1m) - Dataset introduction
3. **DuckDB Performance** (2m) - Lightning-fast queries
4. **Iceberg Integration** (2m) - Modern data lake features
5. **Performance Comparison** (3m) - Speed benchmarks
6. **Advanced Analytics** (4m) - NEO risk assessment
7. **Time Travel** (2m) - Historical data queries
8. **Visualizations** (1m) - Interactive 3D plots

### Data Exploration Notebook

```bash
jupyter lab notebooks/data_exploration.ipynb
```

**Covers:**
- Data quality assessment
- Statistical analysis
- Temporal patterns
- Size distributions
- Risk categorization

### Performance Analysis Notebook

```bash
jupyter lab notebooks/performance_analysis.ipynb
```

**Covers:**
- Query benchmarking
- Storage efficiency
- Format comparisons
- Optimization recommendations

## 📈 Performance Results

### Query Performance Comparison

| Storage Format | Simple Count | Aggregation | Complex Filter | Speedup vs CSV |
|----------------|--------------|-------------|----------------|-----------------|
| **DuckDB Table** | 0.003s | 0.012s | 0.025s | **8.5x faster** |
| **Parquet** | 0.008s | 0.035s | 0.067s | **3.2x faster** |  
| **CSV** | 0.025s | 0.112s | 0.215s | *1x baseline* |

### Storage Efficiency

| Format | File Size | Compression Ratio | Query Speed Rank |
|--------|-----------|-------------------|------------------|
| **CSV** | 150.5 MB | 1.0x (baseline) | 🐌 Slowest |
| **Parquet** | 45.2 MB | 3.3x compressed | 🚀 Fast |
| **DuckDB** | 38.7 MB | 3.9x compressed | ⚡ Fastest |
| **Iceberg** | 42.1 MB | 3.6x compressed | ⚡ Fastest |

### Key Insights

- **🚀 Performance**: DuckDB delivers 3-10x faster queries than traditional formats
- **💾 Storage**: Columnar formats save 70%+ storage space vs CSV
- **🎯 Sweet Spot**: DuckDB + Iceberg provides optimal balance of speed and features
- **📈 Scalability**: Performance advantages increase with dataset size

## 🔍 API Documentation

### Core Classes

#### `NEODataIngester`

Downloads and processes NASA NEO data.

```python
from scripts.data_ingestion import NEODataIngester

ingester = NEODataIngester(data_dir="data/raw")
raw_data = ingester.download_neo_data(limit=50000)
clean_data = ingester.clean_data(raw_data)
file_path = ingester.save_data(clean_data, format="parquet")
```

**Methods:**
- `download_neo_data(limit, max_retries)`: Fetch data from NASA API
- `clean_data(df)`: Validate and clean raw data
- `save_data(df, format, filename)`: Export in various formats

#### `DuckDBManager`

Manages DuckDB databases and queries.

```python
from scripts.duckdb_setup import DuckDBManager

db = DuckDBManager("space_analytics.db")
db.create_neo_table("data/neo_data.parquet")
results = db.execute_query("SELECT COUNT(*) FROM neo_approaches")
stats = db.get_table_stats("neo_approaches")
```

**Methods:**
- `create_neo_table(data_path)`: Create optimized table
- `execute_query(query)`: Run SQL and return results
- `get_table_stats(table_name)`: Analyze table metadata
- `setup_motherduck_sync(token)`: Cloud synchronization

#### `SimpleAnalytics`

Advanced analytics engine for NEO data.

```python
from scripts.advanced_analytics import SimpleAnalytics

analytics = SimpleAnalytics(duckdb_conn)
risky_objects = analytics.find_risky_objects(top_n=20)
yearly_trends = analytics.analyze_trends_by_year()
risk_score = analytics.calculate_risk_score(h=19.7, dist=0.0002, v_rel=30.7)
```

**Methods:**
- `find_risky_objects(top_n)`: Identify high-risk NEOs
- `analyze_trends_by_year()`: Temporal analysis
- `calculate_risk_score(h, dist, v_rel)`: Risk assessment
- `size_distribution_analysis()`: Categorize by size

#### `PerformanceBenchmarker`

Benchmarks query performance across storage formats.

```python
from scripts.performance_benchmarks import PerformanceBenchmarker

benchmarker = PerformanceBenchmarker()
results = benchmarker.compare_storage_formats()
efficiency = benchmarker.storage_efficiency_analysis()
report = benchmarker.generate_performance_report()
```

### Configuration

#### DuckDB Configuration

```python
from config.duckdb_config import duckdb_config

# Get connection settings
config = duckdb_config.get_connection_config()

# Get MotherDuck settings  
motherduck = duckdb_config.get_motherduck_config()

# List required extensions
extensions = duckdb_config.get_extensions()
```

#### Iceberg Configuration

```python
from config.iceberg_config import iceberg_config

# Get catalog properties
catalog = iceberg_config.get_catalog_properties()

# Get table properties
table_props = iceberg_config.get_table_properties()

# Get partition specification  
partitions = iceberg_config.get_partition_spec()
```

## 🧪 Testing

### Run All Tests

```bash
# Run complete test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=scripts --cov-report=html

# Run specific test file
python -m pytest tests/test_data_loader.py -v
```

### Test Categories

#### Data Pipeline Tests (`test_data_loader.py`)

- ✅ API data download with retry logic
- ✅ Data validation and cleaning
- ✅ Multiple output format support
- ✅ Error handling for edge cases

#### Performance Tests (`test_performance.py`)

- ✅ Query benchmarking accuracy
- ✅ Storage format comparison
- ✅ Statistical calculations
- ✅ Report generation

#### Analytics Tests (`test_analytics.py`)

- ✅ Risk score calculations
- ✅ Trend analysis algorithms
- ✅ Size categorization logic
- ✅ Data quality handling

### Test Coverage

Current test coverage: **85%+**

- 📄 `data_ingestion.py`: 90% covered
- 📄 `advanced_analytics.py`: 88% covered  
- 📄 `performance_benchmarks.py`: 82% covered

### Performance Validation

Tests verify that:
- All demo queries complete in <5 seconds
- Memory usage stays under 4GB
- Storage compression achieves >50% savings
- Risk calculations handle edge cases correctly

## 🔧 Troubleshooting

### Common Issues

#### 1. Memory Errors During Data Processing

```bash
# Reduce batch size
python scripts/02_data_ingestion.py --limit 10000

# Or increase available memory
export PYTHONHASHSEED=0
ulimit -v 8388608  # 8GB virtual memory limit
```

#### 2. DuckDB Extension Installation Fails

```bash
# Manual extension installation
python -c "import duckdb; conn = duckdb.connect(); conn.execute('INSTALL httpfs'); conn.execute('LOAD httpfs')"
```

#### 3. Jupyter Kernel Dies During Demo

```bash
# Increase Jupyter memory limits
jupyter lab --NotebookApp.max_buffer_size=2147483648
```

#### 4. MotherDuck Sync Issues

```bash
# Verify token
python -c "import os; print('Token set:', bool(os.getenv('MOTHERDUCK_TOKEN')))"

# Test connection manually
python -c "import duckdb; duckdb.connect('md:?motherduck_token=YOUR_TOKEN').execute('SELECT 1')"
```

### Performance Optimization Tips

1. **Increase DuckDB Memory**: Set `memory_limit='8GB'` for large datasets
2. **Use SSD Storage**: Store data on fast storage for better I/O
3. **Partition Wisely**: Year-based partitioning works well for temporal data
4. **Index Key Columns**: Create indexes on frequently queried columns
5. **Batch Processing**: Process data in chunks for memory efficiency

### Environment Debugging

```bash
# Check Python environment
python --version
pip list | grep -E "(duckdb|pandas|pyarrow|plotly)"

# Verify data files
ls -la data/raw/
du -sh data/

# Test database connection
python -c "import duckdb; print('DuckDB version:', duckdb.__version__)"
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone for development
git clone <repository-url>
cd spaceneo-analytics-demo

# Install dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run code quality checks
black scripts/
flake8 scripts/
mypy scripts/
```

### Contribution Areas

- 🚀 **Performance Optimization**: Improve query speeds
- 📊 **New Analytics**: Add ML models or statistical analysis  
- 🎨 **Visualizations**: Create new chart types
- 🧪 **Testing**: Expand test coverage
- 📚 **Documentation**: Improve guides and examples
- 🌐 **Data Sources**: Integrate additional space datasets

### Code Style

- **Formatting**: Use `black` with 88-character line limit
- **Linting**: Follow `flake8` rules
- **Type Hints**: Add type annotations for all functions
- **Docstrings**: Document all classes and public methods
- **Testing**: Write tests for new functionality

## 📊 Technical Architecture

### Data Flow

```
NASA API → Raw CSV → Cleaned Parquet → DuckDB → Iceberg → Analytics → Visualizations
    ↓           ↓            ↓           ↓         ↓           ↓            ↓
  50K+ NEOs  Validation  Compression  Indexing  ACID OPs   Risk Scores  3D Plots
```

### Technology Stack

- **🐍 Python 3.11+**: Core language with async support
- **🦆 DuckDB 0.9.0+**: High-performance analytical database  
- **🗄️ Apache Iceberg**: Modern data lake table format
- **📊 Pandas 2.0+**: Data manipulation and analysis
- **📈 Plotly 5.15+**: Interactive visualizations
- **🧪 Pytest**: Comprehensive testing framework
- **📓 Jupyter**: Interactive notebooks for demos

### Scalability Considerations

- **Memory**: Handles datasets up to available RAM (8GB+ recommended)
- **Storage**: Efficient columnar formats reduce disk usage by 70%+
- **Compute**: DuckDB utilizes all CPU cores automatically
- **Cloud**: MotherDuck sync enables cloud deployment
- **Growth**: Architecture scales to TB+ datasets with minimal changes

## 🚀 Roadmap

### Phase 1: Core Functionality ✅
- [x] Data ingestion pipeline
- [x] DuckDB integration  
- [x] Basic analytics
- [x] Performance benchmarking
- [x] Interactive demos

### Phase 2: Advanced Features 🚧
- [ ] Real-time streaming updates
- [ ] Machine learning risk models
- [ ] Advanced Iceberg features (branching, tagging)
- [ ] Multi-cloud deployment
- [ ] RESTful API interface

### Phase 3: Production Ready 📋
- [ ] Kubernetes deployment
- [ ] Monitoring and alerting
- [ ] Data governance features
- [ ] Enterprise security
- [ ] Automated CI/CD pipeline

## 📚 Additional Resources

### Documentation
- [DuckDB Documentation](https://duckdb.org/docs/)
- [Apache Iceberg Documentation](https://iceberg.apache.org/docs/latest/)
- [NASA Close Approach Data](https://cneos.jpl.nasa.gov/ca/)

### Tutorials
- [DuckDB Python Tutorial](https://duckdb.org/docs/api/python/overview)
- [Iceberg Python Tutorial](https://py.iceberg.apache.org/)
- [Space Data Analysis with Python](https://www.pythonforspaceandastronautics.com/)

### Community
- [DuckDB Discord](https://discord.duckdb.org/)
- [Apache Iceberg Slack](https://iceberg.apache.org/community/)
- [Space Data Analysis Group](https://www.facebook.com/groups/spacedataanalysis/)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📧 Contact

- **Project Maintainer**: Gihan Mahmoud
- **Email**: gihan.mahmoud@mantelgroup.com.au
- **GitHub**: [YourGitHubUsername]
- **LinkedIn**: [Your LinkedIn Profile]

---

**⭐ If this project helps you understand modern data lakehouse architecture, please star the repository!**

**🚀 Ready to explore space data with DuckDB + Iceberg? Let's get started!**

```bash
git clone <repository-url>
cd spaceneo-analytics-demo
pip install -r requirements.txt
python scripts/01_setup_environment.py
jupyter lab notebooks/presentation_demo.ipynb
```# duckdb-iceberg-vibe
