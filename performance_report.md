# Space Analytics Performance Benchmark Report
Generated: 2025-09-17 12:18:05

## Executive Summary

### Storage Format Performance Comparison

| Format | Avg Query Time (s) | Queries Tested | Performance Rank |
|--------|-------------------|----------------|------------------|
| iceberg | 0.0019 | 4 | #1 |
| duckdb_table | 0.0032 | 5 | #2 |
| parquet | 0.0034 | 5 | #3 |
| csv | 0.1520 | 5 | #4 |

## Detailed Results

### CSV Format

**Simple Count:**
- Mean: 0.1479s
- Median: 0.1480s
- Min: 0.1475s
- Max: 0.1484s
- Rows: 1

**Filtered Count:**
- Mean: 0.1468s
- Median: 0.1468s
- Min: 0.1461s
- Max: 0.1476s
- Rows: 1

**Aggregation:**
- Mean: 0.1497s
- Median: 0.1496s
- Min: 0.1484s
- Max: 0.1511s
- Rows: 10

**Complex Filtering:**
- Mean: 0.1542s
- Median: 0.1544s
- Min: 0.1530s
- Max: 0.1552s
- Rows: 50

**Analytical Window:**
- Mean: 0.1612s
- Median: 0.1605s
- Min: 0.1598s
- Max: 0.1634s
- Rows: 25

### PARQUET Format

**Simple Count:**
- Mean: 0.0002s
- Median: 0.0002s
- Min: 0.0002s
- Max: 0.0003s
- Rows: 1

**Filtered Count:**
- Mean: 0.0006s
- Median: 0.0005s
- Min: 0.0005s
- Max: 0.0007s
- Rows: 1

**Aggregation:**
- Mean: 0.0012s
- Median: 0.0012s
- Min: 0.0012s
- Max: 0.0012s
- Rows: 10

**Complex Filtering:**
- Mean: 0.0036s
- Median: 0.0036s
- Min: 0.0035s
- Max: 0.0038s
- Rows: 50

**Analytical Window:**
- Mean: 0.0116s
- Median: 0.0116s
- Min: 0.0113s
- Max: 0.0119s
- Rows: 25

### DUCKDB_TABLE Format

**Simple Count:**
- Mean: 0.0001s
- Median: 0.0001s
- Min: 0.0001s
- Max: 0.0002s
- Rows: 1

**Filtered Count:**
- Mean: 0.0005s
- Median: 0.0005s
- Min: 0.0005s
- Max: 0.0006s
- Rows: 1

**Aggregation:**
- Mean: 0.0008s
- Median: 0.0008s
- Min: 0.0008s
- Max: 0.0009s
- Rows: 10

**Complex Filtering:**
- Mean: 0.0031s
- Median: 0.0030s
- Min: 0.0029s
- Max: 0.0033s
- Rows: 50

**Analytical Window:**
- Mean: 0.0114s
- Median: 0.0112s
- Min: 0.0107s
- Max: 0.0122s
- Rows: 25

### ICEBERG Format

**Simple Count:**
- Mean: 0.0015s
- Median: 0.0014s
- Min: 0.0012s
- Max: 0.0021s
- Rows: 1

**Filtered Count:**
- Mean: 0.0012s
- Median: 0.0012s
- Min: 0.0012s
- Max: 0.0013s
- Rows: 1

**Aggregation:** ERROR
- Invalid Input Error: Perfect hash aggregate: aggregate group 18446744060824649754 exceeded total groups 32. This likely means that the statistics in your data source are corrupt.
* PRAGMA disable_optimizer to disable optimizations that rely on correct statistics

**Complex Filtering:**
- Mean: 0.0017s
- Median: 0.0017s
- Min: 0.0017s
- Max: 0.0018s
- Rows: 50

**Analytical Window:**
- Mean: 0.0031s
- Median: 0.0031s
- Min: 0.0029s
- Max: 0.0032s
- Rows: 25

## Storage Efficiency Analysis

### File Sizes

- **csv**: 8.02 MB
- **parquet**: 3.92 MB
- **iceberg**: 4.87 MB

### Compression Ratios (vs CSV)

- **parquet**: 2.05x compression
- **iceberg**: 1.65x compression
