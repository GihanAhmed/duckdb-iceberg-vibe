"""
Advanced Analytics for Space Analytics Demo

This script provides advanced analytics capabilities including risk assessment,
trend analysis, and statistical computations for Near-Earth Object data.
"""
# pylint: disable=broad-exception-caught
# pylint: disable=line-too-long

import logging
from pathlib import Path
from typing import Dict, Optional, Any
import json
from datetime import datetime
import duckdb
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleAnalytics:
    """
    Provides advanced analytics capabilities for Near-Earth Object data.

    Implements risk assessment, trend analysis, and statistical computations
    using simple, interpretable algorithms suitable for presentation demos.
    """

    def __init__(self, duckdb_conn: Optional[duckdb.DuckDBPyConnection] = None):
        """Initialize analytics engine with database connection."""
        if duckdb_conn:
            self.conn = duckdb_conn
        else:
            self._setup_connection()

        self.analytics_results = {}

    def _setup_connection(self) -> None:
        """Setup database connection."""
        try:
            if Path("space_analytics.db").exists():
                self.conn = duckdb.connect("space_analytics.db")
                logger.info("✓ Connected to space analytics database")
            else:
                logger.warning("Database not found, using in-memory connection")
                self.conn = duckdb.connect(":memory:")

                # Try to load sample data for demo
                self._load_sample_data()

        except Exception as e:
            logger.error("Failed to setup database connection: %s", e)
            raise

    def _load_sample_data(self) -> None:
        """Load sample data if main database not available."""
        data_files = list(Path("data/raw").glob("neo_data_*.csv"))
        if data_files:
            data_file = data_files[0]
            logger.info("Loading sample data from %s", data_file)

            try:
                self.conn.execute(f"""
                    CREATE TABLE neo_approaches AS
                    SELECT * FROM read_csv_auto('{data_file}')
                """)
                logger.info("✓ Sample data loaded successfully")
            except Exception as e:
                logger.warning("Failed to load sample data: %s", e)

    def calculate_risk_score(self, h: float, dist: float, v_rel: float) -> float:
        """
        Calculate simple risk score for Near-Earth Objects.

        Simple scoring algorithm:
        - Size factor: Larger objects (lower H magnitude) are riskier
        - Distance factor: Closer approaches are riskier
        - Velocity factor: Faster objects are riskier

        Args:
            h: Absolute magnitude (smaller = larger object)
            dist: Distance in AU
            v_rel: Relative velocity in km/s

        Returns:
            Risk score from 0-100 (higher = more risk)
        """
        if any(pd.isna(x) for x in [h, dist, v_rel]):
            return 0.0

        # Size component (0-40 points): H=30 is small, H=10 is huge
        size_score = max(0, 30 - h) * 1.33  # Max 40 points for H=0

        # Distance component (0-40 points): <0.1 AU is very close
        distance_score = max(0, 0.1 - dist) * 400  # Max 40 points for dist=0

        # Speed component (0-20 points): Up to 60 km/s
        speed_score = min(v_rel, 60) * 0.33  # Max 20 points for 60+ km/s

        total_score = min(size_score + distance_score + speed_score, 100)
        return round(total_score, 2)

    def find_risky_objects(self, top_n: int = 20) -> pd.DataFrame:
        """
        Find the most risky Near-Earth Objects.

        Args:
            top_n: Number of top risky objects to return

        Returns:
            DataFrame with risky objects and their risk scores
        """
        logger.info("Finding top %d risky objects...", top_n)

        try:
            query = f"""
            SELECT
                des,
                fullname,
                h,
                ROUND(dist, 6) as distance_au,
                ROUND(v_rel, 2) as velocity_kms,
                cd as approach_date,
                ROUND(
                    GREATEST(0, 30 - h) * 1.33 +
                    GREATEST(0, 0.1 - dist) * 400 +
                    LEAST(v_rel, 60) * 0.33,
                    2
                ) as risk_score,
                CASE
                    WHEN h < 18 AND dist < 0.05 THEN 'VERY_HIGH'
                    WHEN h < 20 AND dist < 0.1 THEN 'HIGH'
                    WHEN h < 25 AND dist < 0.2 THEN 'MEDIUM'
                    ELSE 'LOW'
                END as risk_category
            FROM neo_approaches
            WHERE h IS NOT NULL AND dist IS NOT NULL AND v_rel IS NOT NULL
            ORDER BY risk_score DESC
            LIMIT {top_n}
            """

            result_df = self.conn.execute(query).df()

            if not result_df.empty:
                logger.info("✓ Found %d risky objects", len(result_df))
                logger.info("  Highest risk score: %.2f", result_df['risk_score'].max())
                logger.info("  Risk categories: %s", result_df['risk_category'].value_counts().to_dict())

                self.analytics_results['risky_objects'] = result_df.to_dict('records')
            else:
                logger.warning("No risky objects found")

            return result_df

        except Exception as e:
            logger.error("Failed to find risky objects: %s", e)
            return pd.DataFrame()

    def analyze_trends_by_year(self) -> pd.DataFrame:
        """
        Analyze discovery and approach trends over time.

        Returns:
            DataFrame with yearly trend analysis
        """
        logger.info("Analyzing yearly trends...")

        try:
            query = """
            SELECT
                approach_year as year,
                COUNT(*) as total_approaches,
                COUNT(DISTINCT des) as unique_objects,
                ROUND(AVG(dist), 6) as avg_distance_au,
                ROUND(MIN(dist), 6) as closest_approach_au,
                ROUND(MAX(dist), 6) as farthest_approach_au,
                ROUND(AVG(v_rel), 2) as avg_velocity_kms,
                ROUND(AVG(h), 2) as avg_magnitude,
                COUNT(CASE WHEN h < 22 THEN 1 END) as large_objects,
                COUNT(CASE WHEN dist < 0.05 THEN 1 END) as close_approaches
            FROM neo_approaches
            WHERE approach_year IS NOT NULL AND approach_year BETWEEN 1900 AND 2030
            GROUP BY approach_year
            ORDER BY approach_year DESC
            LIMIT 25
            """

            result_df = self.conn.execute(query).df()

            if not result_df.empty:
                logger.info("✓ Analyzed trends for %d years", len(result_df))

                # Calculate trend statistics
                recent_years = result_df.head(10)
                avg_approaches_recent = recent_years['total_approaches'].mean()

                logger.info("  Recent years avg approaches: %.1f", avg_approaches_recent)
                logger.info("  Peak year: %d", result_df.loc[result_df['total_approaches'].idxmax(), 'year'])

                self.analytics_results['yearly_trends'] = result_df.to_dict('records')
            else:
                logger.warning("No yearly trend data found")

            return result_df

        except Exception as e:
            logger.error("Failed to analyze yearly trends: %s", e)
            return pd.DataFrame()

    def size_distribution_analysis(self) -> pd.DataFrame:
        """
        Analyze size distribution of Near-Earth Objects.

        Returns:
            DataFrame with size category analysis
        """
        logger.info("Analyzing size distribution...")

        try:
            query = """
            SELECT
                CASE
                    WHEN h < 18 THEN 'Very Large (>1km)'
                    WHEN h < 22 THEN 'Large (140m-1km)'
                    WHEN h < 25 THEN 'Medium (30-140m)'
                    ELSE 'Small (<30m)'
                END as size_category,
                COUNT(*) as object_count,
                ROUND(MIN(dist), 6) as closest_ever_au,
                ROUND(AVG(dist), 6) as avg_distance_au,
                ROUND(AVG(v_rel), 2) as avg_velocity_kms,
                ROUND(MIN(h), 2) as brightest_magnitude,
                ROUND(MAX(h), 2) as dimmest_magnitude,
                COUNT(CASE WHEN dist < 0.05 THEN 1 END) as close_approaches
            FROM neo_approaches
            WHERE h IS NOT NULL
            GROUP BY size_category
            ORDER BY MIN(h)
            """

            result_df = self.conn.execute(query).df()

            if not result_df.empty:
                logger.info("✓ Analyzed %d size categories", len(result_df))

                total_objects = result_df['object_count'].sum()
                for _, row in result_df.iterrows():
                    percentage = (row['object_count'] / total_objects) * 100
                    logger.info("  %s: %d (%.1f%%)", row['size_category'], row['object_count'], percentage)

                self.analytics_results['size_distribution'] = result_df.to_dict('records')
            else:
                logger.warning("No size distribution data found")

            return result_df

        except Exception as e:
            logger.error("Failed to analyze size distribution: %s", e)
            return pd.DataFrame()

    def velocity_analysis(self) -> pd.DataFrame:
        """
        Analyze velocity characteristics of Near-Earth Objects.

        Returns:
            DataFrame with velocity analysis
        """
        logger.info("Analyzing velocity characteristics...")

        try:
            query = """
            SELECT
                CASE
                    WHEN v_rel < 10 THEN 'Slow (<10 km/s)'
                    WHEN v_rel < 20 THEN 'Moderate (10-20 km/s)'
                    WHEN v_rel < 30 THEN 'Fast (20-30 km/s)'
                    ELSE 'Very Fast (>30 km/s)'
                END as velocity_category,
                COUNT(*) as object_count,
                ROUND(MIN(v_rel), 2) as min_velocity,
                ROUND(MAX(v_rel), 2) as max_velocity,
                ROUND(AVG(v_rel), 2) as avg_velocity,
                ROUND(AVG(dist), 6) as avg_distance_au,
                ROUND(AVG(h), 2) as avg_magnitude,
                COUNT(CASE WHEN dist < 0.05 THEN 1 END) as close_approaches
            FROM neo_approaches
            WHERE v_rel IS NOT NULL AND v_rel > 0
            GROUP BY velocity_category
            ORDER BY MIN(v_rel)
            """

            result_df = self.conn.execute(query).df()

            if not result_df.empty:
                logger.info("✓ Analyzed %d velocity categories", len(result_df))

                self.analytics_results['velocity_analysis'] = result_df.to_dict('records')
            else:
                logger.warning("No velocity analysis data found")

            return result_df

        except Exception as e:
            logger.error("Failed to analyze velocity characteristics: %s", e)
            return pd.DataFrame()

    def close_approach_analysis(self) -> Dict[str, Any]:
        """
        Analyze close approach patterns and statistics.

        Returns:
            Dictionary with close approach analysis results
        """
        logger.info("Analyzing close approach patterns...")

        analysis_results = {}

        try:
            # Distance statistics
            distance_stats_query = """
            SELECT
                COUNT(*) as total_approaches,
                ROUND(MIN(dist), 8) as closest_ever,
                ROUND(MAX(dist), 6) as farthest,
                ROUND(AVG(dist), 6) as mean_distance,
                ROUND(MEDIAN(dist), 6) as median_distance,
                ROUND(STDDEV(dist), 6) as std_distance,
                COUNT(CASE WHEN dist < 0.01 THEN 1 END) as very_close_approaches,
                COUNT(CASE WHEN dist < 0.05 THEN 1 END) as close_approaches,
                COUNT(CASE WHEN dist < 0.1 THEN 1 END) as notable_approaches
            FROM neo_approaches
            WHERE dist IS NOT NULL
            """

            distance_stats = self.conn.execute(distance_stats_query).df().iloc[0].to_dict()
            analysis_results['distance_statistics'] = distance_stats

            # Monthly distribution
            monthly_query = """
            SELECT
                EXTRACT(month FROM CAST(cd AS DATE)) as month,
                COUNT(*) as approach_count,
                ROUND(AVG(dist), 6) as avg_distance,
                COUNT(CASE WHEN dist < 0.05 THEN 1 END) as close_approaches
            FROM neo_approaches
            WHERE cd IS NOT NULL
            GROUP BY month
            ORDER BY month
            """

            monthly_df = self.conn.execute(monthly_query).df()
            if not monthly_df.empty:
                analysis_results['monthly_distribution'] = monthly_df.to_dict('records')

            # Record holders
            records_query = """
            SELECT
                'Closest Approach' as record_type,
                des,
                fullname,
                ROUND(dist, 8) as distance_au,
                cd as date,
                ROUND(v_rel, 2) as velocity_kms
            FROM neo_approaches
            WHERE dist = (SELECT MIN(dist) FROM neo_approaches WHERE dist IS NOT NULL)

            UNION ALL

            SELECT
                'Fastest Object' as record_type,
                des,
                fullname,
                ROUND(dist, 6) as distance_au,
                cd as date,
                ROUND(v_rel, 2) as velocity_kms
            FROM neo_approaches
            WHERE v_rel = (SELECT MAX(v_rel) FROM neo_approaches WHERE v_rel IS NOT NULL)

            UNION ALL

            SELECT
                'Largest Object' as record_type,
                des,
                fullname,
                ROUND(dist, 6) as distance_au,
                cd as date,
                h as velocity_kms
            FROM neo_approaches
            WHERE h = (SELECT MIN(h) FROM neo_approaches WHERE h IS NOT NULL)
            """

            records_df = self.conn.execute(records_query).df()
            if not records_df.empty:
                analysis_results['record_holders'] = records_df.to_dict('records')

            logger.info("✓ Close approach analysis completed")
            logger.info("  Total approaches: %s", distance_stats.get('total_approaches', 'N/A'))
            logger.info("  Closest ever: %s AU", distance_stats.get('closest_ever', 'N/A'))
            logger.info("  Very close approaches (<0.01 AU): %s", distance_stats.get('very_close_approaches', 'N/A'))

            self.analytics_results['close_approach_analysis'] = analysis_results
            return analysis_results

        except Exception as e:
            logger.error("Failed to analyze close approaches: %s", e)
            return {"error": str(e)}

    def generate_analytics_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive analytics summary.

        Returns:
            Dictionary with complete analytics summary
        """
        logger.info("Generating analytics summary...")

        summary = {
            "generated": datetime.now().isoformat(),
            "analytics_modules": [],
            "key_findings": [],
            "data_quality": {}
        }

        try:
            # Data quality assessment
            quality_query = """
            SELECT
                COUNT(*) as total_records,
                COUNT(CASE WHEN des IS NOT NULL THEN 1 END) as valid_designations,
                COUNT(CASE WHEN h IS NOT NULL THEN 1 END) as valid_magnitudes,
                COUNT(CASE WHEN dist IS NOT NULL THEN 1 END) as valid_distances,
                COUNT(CASE WHEN v_rel IS NOT NULL THEN 1 END) as valid_velocities,
                COUNT(CASE WHEN cd IS NOT NULL THEN 1 END) as valid_dates,
                MIN(approach_year) as earliest_year,
                MAX(approach_year) as latest_year
            FROM neo_approaches
            """

            quality_stats = self.conn.execute(quality_query).df().iloc[0].to_dict()
            summary["data_quality"] = quality_stats

            # Compile results from all analytics modules
            if self.analytics_results:
                summary["results"] = self.analytics_results

                # Extract key findings
                findings = []

                if 'risky_objects' in self.analytics_results:
                    risky_count = len(self.analytics_results['risky_objects'])
                    if risky_count > 0:
                        highest_risk = max(obj['risk_score'] for obj in self.analytics_results['risky_objects'])
                        findings.append(f"Identified {risky_count} high-risk objects (max risk score: {highest_risk})")

                if 'yearly_trends' in self.analytics_results:
                    trends = self.analytics_results['yearly_trends']
                    if trends:
                        recent_year = trends[0]
                        findings.append(f"Most recent year: {recent_year['year']} with {recent_year['total_approaches']} approaches")

                if 'size_distribution' in self.analytics_results:
                    size_data = self.analytics_results['size_distribution']
                    total_objects = sum(cat['object_count'] for cat in size_data)
                    findings.append(f"Analyzed {total_objects} objects across {len(size_data)} size categories")

                if 'close_approach_analysis' in self.analytics_results:
                    close_data = self.analytics_results['close_approach_analysis']
                    if 'distance_statistics' in close_data:
                        closest = close_data['distance_statistics'].get('closest_ever')
                        if closest:
                            findings.append(f"Closest recorded approach: {closest} AU")

                summary["key_findings"] = findings

            # Add module completion status
            modules = [
                'risky_objects', 'yearly_trends', 'size_distribution',
                'velocity_analysis', 'close_approach_analysis'
            ]

            for module in modules:
                status = "completed" if module in self.analytics_results else "not_run"
                summary["analytics_modules"].append({"module": module, "status": status})

            logger.info("✓ Analytics summary generated")
            completed_count = len([m for m in summary['analytics_modules'] if m['status'] == 'completed'])
            logger.info("  Completed modules: %d", completed_count)
            logger.info("  Key findings: %d", len(summary['key_findings']))

            return summary

        except Exception as e:
            logger.error("Failed to generate analytics summary: %s", e)
            return {"error": str(e)}

    def save_results(self, output_file: str = "analytics_results.json") -> str:
        """Save analytics results to file."""
        try:
            summary = self.generate_analytics_summary()

            # Determine output format based on file extension
            output_path = Path(output_file)

            if output_path.suffix.lower() == '.md':
                self._save_as_markdown(summary, output_path)
            else:
                # Default to JSON
                with open(output_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)

            logger.info("✓ Analytics results saved to %s", output_path)
            return str(output_path)

        except Exception as e:
            logger.error("Failed to save results: %s", e)
            return ""

    def _save_as_markdown(self, summary: Dict[str, Any], output_path: Path) -> None:
        """Save analytics results as markdown report."""
        from datetime import datetime

        with open(output_path, 'w') as f:
            # Header
            f.write("# Space Analytics Advanced Analytics Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("### Analytics Modules Status\n\n")
            f.write("| Module | Status |\n")
            f.write("|--------|--------|\n")

            for module in summary.get('analytics_modules', []):
                status_icon = "✅" if module['status'] == 'completed' else "❌"
                f.write(f"| {module['module'].replace('_', ' ').title()} | {status_icon} {module['status']} |\n")

            # Key Findings
            if summary.get('key_findings'):
                f.write("\n### Key Findings\n\n")
                for finding in summary['key_findings']:
                    f.write(f"- {finding}\n")

            # Data Quality Assessment
            if summary.get('data_quality'):
                f.write("\n## Data Quality Assessment\n\n")
                quality = summary['data_quality']
                f.write(f"- **Total Records**: {quality.get('total_records', 'N/A'):,}\n")
                f.write(f"- **Valid Designations**: {quality.get('valid_designations', 'N/A'):,}\n")
                f.write(f"- **Valid Magnitudes**: {quality.get('valid_magnitudes', 'N/A'):,}\n")
                f.write(f"- **Valid Distances**: {quality.get('valid_distances', 'N/A'):,}\n")
                f.write(f"- **Valid Velocities**: {quality.get('valid_velocities', 'N/A'):,}\n")
                f.write(f"- **Year Range**: {quality.get('earliest_year', 'N/A')} - {quality.get('latest_year', 'N/A')}\n")

            # Detailed Results
            if summary.get('results'):
                f.write("\n## Detailed Analysis Results\n\n")
                results = summary['results']

                # Risk Objects
                if 'risky_objects' in results:
                    f.write("### High-Risk Objects\n\n")
                    risky_objects = results['risky_objects']
                    if risky_objects:
                        f.write("| Designation | Name | Risk Score | Distance (AU) | Velocity (km/s) |\n")
                        f.write("|-------------|------|------------|---------------|----------------|\n")
                        for obj in risky_objects[:10]:  # Top 10
                            f.write(f"| {obj.get('des', 'N/A')} | {obj.get('fullname', 'N/A')} | {obj.get('risk_score', 0):.2f} | {obj.get('distance_au', 0):.6f} | {obj.get('velocity_kms', 0):.2f} |\n")

                # Size Distribution
                if 'size_distribution' in results:
                    f.write("\n### Size Distribution Analysis\n\n")
                    size_data = results['size_distribution']
                    if size_data:
                        f.write("| Size Category | Object Count | Avg Distance (AU) | Avg Velocity (km/s) |\n")
                        f.write("|---------------|--------------|-------------------|--------------------|\n")
                        for category in size_data:
                            f.write(f"| {category.get('size_category', 'N/A')} | {category.get('object_count', 0):,} | {category.get('avg_distance_au', 0):.6f} | {category.get('avg_velocity_kms', 0):.2f} |\n")

                # Yearly Trends (Top 10 most recent)
                if 'yearly_trends' in results:
                    f.write("\n### Yearly Trends (Most Recent)\n\n")
                    yearly_data = results['yearly_trends']
                    if yearly_data:
                        f.write("| Year | Total Approaches | Unique Objects | Avg Distance (AU) | Closest Approach (AU) |\n")
                        f.write("|------|------------------|----------------|-------------------|----------------------|\n")
                        for year_data in yearly_data[:10]:  # Top 10
                            f.write(f"| {year_data.get('year', 'N/A')} | {year_data.get('total_approaches', 0):,} | {year_data.get('unique_objects', 0):,} | {year_data.get('avg_distance_au', 0):.6f} | {year_data.get('closest_approach_au', 0):.6f} |\n")

                # Close Approach Analysis
                if 'close_approach_analysis' in results:
                    close_data = results['close_approach_analysis']
                    f.write("\n### Close Approach Analysis\n\n")

                    if 'distance_statistics' in close_data:
                        stats = close_data['distance_statistics']
                        f.write("#### Distance Statistics\n\n")
                        f.write(f"- **Total Approaches**: {stats.get('total_approaches', 'N/A'):,}\n")
                        f.write(f"- **Closest Ever**: {stats.get('closest_ever', 'N/A')} AU\n")
                        f.write(f"- **Farthest**: {stats.get('farthest_approach', 'N/A')} AU\n")
                        f.write(f"- **Average Distance**: {stats.get('mean_distance', 'N/A')} AU\n")
                        f.write(f"- **Very Close Approaches** (<0.01 AU): {stats.get('very_close_approaches', 'N/A'):,}\n")
                        f.write(f"- **Close Approaches** (<0.05 AU): {stats.get('close_approaches', 'N/A'):,}\n")

                    if 'record_holders' in close_data:
                        f.write("\n#### Record Holders\n\n")
                        records = close_data['record_holders']
                        if records:
                            f.write("| Record Type | Designation | Name | Value | Date |\n")
                            f.write("|-------------|-------------|------|-------|------|\n")
                            for record in records:
                                f.write(f"| {record.get('record_type', 'N/A')} | {record.get('des', 'N/A')} | {record.get('fullname', 'N/A')} | {record.get('distance_au', record.get('velocity_kms', 'N/A'))} | {record.get('date', 'N/A')} |\n")

                # Velocity Analysis
                if 'velocity_analysis' in results:
                    f.write("\n### Velocity Analysis\n\n")
                    velocity_data = results['velocity_analysis']
                    if 'velocity_statistics' in velocity_data:
                        vel_stats = velocity_data['velocity_statistics']
                        f.write("#### Velocity Statistics\n\n")
                        f.write(f"- **Average Velocity**: {vel_stats.get('mean_velocity', 'N/A')} km/s\n")
                        f.write(f"- **Median Velocity**: {vel_stats.get('median_velocity', 'N/A')} km/s\n")
                        f.write(f"- **Fastest Object**: {vel_stats.get('max_velocity', 'N/A')} km/s\n")
                        f.write(f"- **Slowest Object**: {vel_stats.get('min_velocity', 'N/A')} km/s\n")

            f.write(f"\n---\n*Report generated by Space Analytics Advanced Analytics Engine*\n")

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("✓ Analytics database connection closed")


def main():
    """Main execution function for advanced analytics."""
    logger.info("=" * 60)
    logger.info("ADVANCED NEO ANALYTICS")
    logger.info("=" * 60)

    analytics = None

    try:
        # Initialize analytics engine
        analytics = SimpleAnalytics()

        # Run all analytics modules
        logger.info("\nRunning analytics modules...")

        # 1. Risk assessment
        risky_objects = analytics.find_risky_objects(top_n=15)

        # 2. Yearly trends
        yearly_trends = analytics.analyze_trends_by_year()

        # 3. Size distribution
        size_distribution = analytics.size_distribution_analysis()

        # 4. Velocity analysis
        velocity_analysis = analytics.velocity_analysis()

        # 5. Close approach patterns
        close_approach_results = analytics.close_approach_analysis()

        # Generate summary and save results
        results_file = analytics.save_results("analytics_results.md")

        # Display summary
        logger.info("\n%s\nANALYTICS SUMMARY\n%s", "="*60, "="*60)

        if not risky_objects.empty:
            logger.info("High-risk objects identified: %d", len(risky_objects))
            top_risk = risky_objects.iloc[0]
            logger.info("  Highest risk: %s (score: %.2f)", top_risk['des'], top_risk['risk_score'])

        if not yearly_trends.empty:
            logger.info("Years analyzed: %d", len(yearly_trends))
            peak_year = yearly_trends.loc[yearly_trends['total_approaches'].idxmax()]
            logger.info("  Peak activity: %d (%d approaches)",
                        peak_year['year'], peak_year['total_approaches'])

        if not size_distribution.empty:
            logger.info("Size categories: %d", len(size_distribution))
            largest_category = size_distribution.iloc[0]
            logger.info("  Largest objects: %d %s",
                        largest_category['object_count'], largest_category['size_category'])

        if close_approach_results and 'distance_statistics' in close_approach_results:
            stats = close_approach_results['distance_statistics']
            logger.info("Total approaches analyzed: %s", stats.get('total_approaches', 'N/A'))
            logger.info("  Closest approach: %s AU", stats.get('closest_ever', 'N/A'))

        if results_file:
            logger.info("\nResults saved to: %s", results_file)

        logger.info("\n🎉 Advanced analytics completed successfully!")
        logger.info("Next step: python scripts/07_visualization.py")

        return True

    except Exception as e:
        logger.error("❌ Advanced analytics failed: %s", e)
        return False
    finally:
        if analytics:
            analytics.close()


if __name__ == "__main__":
    import sys
    SUCCESS = main()
    sys.exit(0 if SUCCESS else 1)
