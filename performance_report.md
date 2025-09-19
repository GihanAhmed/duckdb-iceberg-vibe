# Space Analytics Performance Benchmark Report
Generated: 2025-09-19 10:38:47

## Executive Summary

### Storage Format Performance Comparison

| Format | Avg Query Time (s) | Queries Tested | Performance Rank |
|--------|-------------------|----------------|------------------|
| duckdb_table | 0.0033 | 5 | #1 |
| parquet | 0.0037 | 5 | #2 |
| iceberg | 0.0044 | 5 | #3 |
| csv | 0.1617 | 5 | #4 |

## Detailed Results

### CSV Format

**Simple Count:**
- Mean: 0.1550s
- Median: 0.1543s
- Min: 0.1536s
- Max: 0.1572s
- Rows: 1

**Filtered Count:**
- Mean: 0.1538s
- Median: 0.1540s
- Min: 0.1527s
- Max: 0.1547s
- Rows: 1

**Aggregation:**
- Mean: 0.1560s
- Median: 0.1565s
- Min: 0.1542s
- Max: 0.1574s
- Rows: 10

**Complex Filtering:**
- Mean: 0.1712s
- Median: 0.1615s
- Min: 0.1573s
- Max: 0.1947s
- Rows: 50

**Analytical Window:**
- Mean: 0.1725s
- Median: 0.1723s
- Min: 0.1722s
- Max: 0.1729s
- Rows: 25

### PARQUET Format

**Simple Count:**
- Mean: 0.0003s
- Median: 0.0002s
- Min: 0.0002s
- Max: 0.0004s
- Rows: 1

**Filtered Count:**
- Mean: 0.0006s
- Median: 0.0006s
- Min: 0.0005s
- Max: 0.0007s
- Rows: 1

**Aggregation:**
- Mean: 0.0012s
- Median: 0.0012s
- Min: 0.0011s
- Max: 0.0012s
- Rows: 10

**Complex Filtering:**
- Mean: 0.0038s
- Median: 0.0038s
- Min: 0.0036s
- Max: 0.0039s
- Rows: 50

**Analytical Window:**
- Mean: 0.0129s
- Median: 0.0130s
- Min: 0.0124s
- Max: 0.0132s
- Rows: 25

### DUCKDB_TABLE Format

**Simple Count:**
- Mean: 0.0001s
- Median: 0.0001s
- Min: 0.0001s
- Max: 0.0002s
- Rows: 1

**Filtered Count:**
- Mean: 0.0006s
- Median: 0.0006s
- Min: 0.0005s
- Max: 0.0007s
- Rows: 1

**Aggregation:**
- Mean: 0.0009s
- Median: 0.0009s
- Min: 0.0009s
- Max: 0.0010s
- Rows: 10

**Complex Filtering:**
- Mean: 0.0024s
- Median: 0.0024s
- Min: 0.0023s
- Max: 0.0024s
- Rows: 50

**Analytical Window:**
- Mean: 0.0124s
- Median: 0.0125s
- Min: 0.0121s
- Max: 0.0126s
- Rows: 25

### ICEBERG Format

**Simple Count:**
- Mean: 0.0002s
- Median: 0.0002s
- Min: 0.0002s
- Max: 0.0002s
- Rows: 1

**Filtered Count:**
- Mean: 0.0007s
- Median: 0.0006s
- Min: 0.0006s
- Max: 0.0007s
- Rows: 1

**Aggregation:**
- Mean: 0.0017s
- Median: 0.0016s
- Min: 0.0016s
- Max: 0.0017s
- Rows: 10

**Complex Filtering:**
- Mean: 0.0061s
- Median: 0.0058s
- Min: 0.0057s
- Max: 0.0067s
- Rows: 50

**Analytical Window:**
- Mean: 0.0135s
- Median: 0.0133s
- Min: 0.0130s
- Max: 0.0143s
- Rows: 25

## Storage Efficiency Analysis

### File Sizes

- **csv**: 8.02 MB
- **parquet**: 3.92 MB
- **iceberg**: 3.51 MB

### Compression Ratios (vs CSV)

- **parquet**: 2.05x compression
- **iceberg**: 2.29x compression
