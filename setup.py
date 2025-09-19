from setuptools import setup, find_packages

setup(
    name="spaceneo-analytics",
    version="0.1.0",
    description="Space Analytics Demo with DuckDB and Apache Iceberg",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "duckdb>=0.9.0",
        "pyiceberg[sql,duckdb]>=0.5.0",
        "plotly>=5.0.0",
        "requests>=2.25.0",
        "pyarrow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
            "jupyterlab>=3.0.0",
        ]
    }
)