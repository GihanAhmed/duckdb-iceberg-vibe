"""
Data Visualization for Space Analytics Demo

This script creates interactive visualizations and charts for the space analytics
demo, including performance comparisons, risk assessments, and discovery timelines.
"""
# pylint: disable=broad-exception-caught
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Third-party imports
try:
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import duckdb
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning("Visualization libraries not available: %s", e)
    VISUALIZATION_AVAILABLE = False


class SpaceDataVisualizer:
    """
    Creates interactive visualizations for the Space Analytics demo.

    Provides functionality for generating performance comparison charts,
    risk assessment dashboards, discovery timelines, and presentation-ready figures.
    """

    def __init__(self, data_source: Optional[Union[str, duckdb.DuckDBPyConnection]] = None):
        """Initialize visualizer with data source."""
        self.data_source = data_source
        self.conn = None
        self.figures = {}
        self._setup_data_connection()
        self._setup_style()

    def _setup_data_connection(self) -> None:
        """Setup data connection for visualization."""
        try:
            if isinstance(self.data_source, duckdb.DuckDBPyConnection):
                self.conn = self.data_source
                logger.info("✓ Using provided DuckDB connection")
            elif isinstance(self.data_source, str) and Path(self.data_source).exists():
                self.conn = duckdb.connect(self.data_source)
                logger.info("✓ Connected to database: %s", self.data_source)
            else:
                # Default connection
                if Path("space_analytics.db").exists():
                    self.conn = duckdb.connect("space_analytics.db")
                    logger.info("✓ Connected to default space analytics database")
                else:
                    self.conn = duckdb.connect(":memory:")
                    logger.warning("Using in-memory database - limited functionality")

        except (duckdb.Error, OSError) as e:
            logger.error("Failed to setup data connection: %s", e)
            self.conn = duckdb.connect(":memory:")

    def _setup_style(self) -> None:
        """Setup consistent styling for all charts."""
        # Set dark space theme for matplotlib
        plt.style.use('dark_background')

        # Enhanced color palette with plasma/cosmic themes
        self.color_palette = [
            '#440154',  # Deep purple
            '#31688e',  # Blue
            '#35b779',  # Green
            '#fde725',  # Yellow
            '#fd7f6f',  # Coral
            '#7eb0d3',  # Light blue
            '#b2e061',  # Light green
            '#bd7ebe',  # Light purple
            '#ffb55a',  # Orange
            '#ffee65'   # Light yellow
        ]

        # Chart configurations from PROJECT_REQUIREMENTS
        self.chart_configs = {
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

        # Set seaborn palette
        sns.set_palette(self.color_palette)

        logger.info("✓ Enhanced visualization styling configured")

    def create_performance_comparison_chart(
        self, benchmark_results: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create interactive performance comparison chart.

        Args:
            benchmark_results: Performance benchmark results dictionary

        Returns:
            Plotly figure with performance comparison
        """
        logger.info("Creating performance comparison chart...")

        try:
            # Load benchmark results if not provided
            if benchmark_results is None:
                # Try JSON first, then fall back to markdown
                json_file = Path("performance_report.json")
                md_file = Path("performance_report.md")

                if json_file.exists():
                    with open(json_file, 'r', encoding='utf-8') as f:
                        benchmark_results = json.load(f)
                elif md_file.exists():
                    logger.info("Loading benchmark results from markdown file")
                    benchmark_results = self._parse_performance_markdown(md_file)
                else:
                    logger.warning("No benchmark results found, creating sample data")
                    benchmark_results = self._create_sample_performance_data()

            # Extract performance data
            performance_data = []

            if isinstance(benchmark_results, dict) and 'format_comparison' in benchmark_results:
                for format_name, results in benchmark_results['format_comparison'].items():
                    if results.get('status') == 'completed':
                        for query_name, query_result in results.items():
                            if (query_name != 'status' and
                                isinstance(query_result, dict) and
                                'mean' in query_result):
                                performance_data.append({
                                    'format': format_name,
                                    'query': query_name.replace('_', ' ').title(),
                                    'mean_time': query_result['mean'],
                                    'min_time': query_result['min'],
                                    'max_time': query_result['max'],
                                    'median_time': query_result['median']
                                })

            if not performance_data:
                logger.warning("No performance data available, creating sample")
                performance_data = [
                    {
                        'format': 'CSV', 'query': 'Simple Count', 'mean_time': 2.5,
                        'min_time': 2.2, 'max_time': 2.8, 'median_time': 2.4
                    },
                    {
                        'format': 'Parquet', 'query': 'Simple Count', 'mean_time': 0.8,
                        'min_time': 0.7, 'max_time': 0.9, 'median_time': 0.8
                    },
                    {
                        'format': 'DuckDB', 'query': 'Simple Count', 'mean_time': 0.3,
                        'min_time': 0.2, 'max_time': 0.4, 'median_time': 0.3
                    },
                    {
                        'format': 'Iceberg', 'query': 'Simple Count', 'mean_time': 0.4,
                        'min_time': 0.3, 'max_time': 0.5, 'median_time': 0.4
                    },
                ]

            df = pd.DataFrame(performance_data)

            # Create subplot figure with enhanced styling
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Query Performance by Format', 'Performance Distribution',
                              'Format Comparison', 'Query Complexity Impact'),
                specs=[[{"type": "bar"}, {"type": "box"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )

            # Apply the configured title and styling
            config = self.chart_configs["performance_chart"]
            fig.update_layout(
                title={
                    "text": config["title"],
                    "x": 0.5,
                    "font": {"size": 20, "color": "white"}
                },
                plot_bgcolor='rgba(0,0,0,0.8)',
                paper_bgcolor='rgba(0,0,0,0.9)',
                font={"color": "white"},
                template='plotly_dark'
            )

            # 1. Bar chart - mean performance by format with plasma colors
            formats = df['format'].unique()
            mean_times = df.groupby('format')['mean_time'].mean()

            # Use enhanced color palette
            colors = self.color_palette[:len(formats)]

            fig.add_trace(
                go.Bar(x=formats, y=mean_times, name='Avg Query Time',
                       marker_color=colors,
                       marker_line=dict(color='rgba(255,255,255,0.3)', width=1),
                       text=[f'{t:.3f}s' for t in mean_times],
                       textposition='auto',
                       textfont=dict(color='white', size=12)),
                row=1, col=1
            )

            # 2. Box plot - performance distribution
            for i, format_name in enumerate(formats):
                format_data = df[df['format'] == format_name]['mean_time']
                color_idx = min(i, len(self.color_palette) - 1)
                fig.add_trace(
                    go.Box(y=format_data, name=format_name,
                           marker_color=self.color_palette[color_idx]),
                    row=1, col=2
                )

            # 3. Grouped bar chart - detailed comparison
            queries = df['query'].unique()
            for i, query in enumerate(queries):
                query_data = df[df['query'] == query]
                color_idx = min(i, len(self.color_palette) - 1)
                fig.add_trace(
                    go.Bar(x=query_data['format'], y=query_data['mean_time'],
                           name=query, marker_color=self.color_palette[color_idx]),
                    row=2, col=1
                )

            # 4. Scatter plot - complexity vs performance
            fig.add_trace(
                go.Scatter(x=df.index, y=df['mean_time'],
                          mode='markers+lines',
                          marker={
                              "size": 10, "color": df['mean_time'],
                              "colorscale": 'Viridis', "showscale": True
                          },
                          text=df['format'] + '<br>' + df['query'],
                          name='Performance Trend'),
                row=2, col=2
            )

            # Update layout
            fig.update_layout(
                title={
                    "text": "DuckDB + Iceberg vs Traditional Storage Performance",
                    "x": 0.5,
                    "font": {"size": 20}
                },
                showlegend=True,
                height=800,
                template='plotly_dark'
            )

            # Update axes labels
            fig.update_xaxes(title_text="Storage Format", row=1, col=1)
            fig.update_yaxes(title_text="Query Time (seconds)", row=1, col=1)
            fig.update_yaxes(title_text="Query Time (seconds)", row=1, col=2)
            fig.update_xaxes(title_text="Storage Format", row=2, col=1)
            fig.update_yaxes(title_text="Query Time (seconds)", row=2, col=1)
            fig.update_xaxes(title_text="Query Index", row=2, col=2)
            fig.update_yaxes(title_text="Query Time (seconds)", row=2, col=2)

            self.figures['performance_comparison'] = fig
            logger.info("✓ Performance comparison chart created")

            return fig

        except (duckdb.Error, ValueError) as e:
            logger.error("Failed to create performance comparison chart: %s", e)
            return go.Figure()

    def create_risk_assessment_dashboard(
        self, risk_data: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create comprehensive risk assessment dashboard.

        Args:
            risk_data: DataFrame with risk assessment data

        Returns:
            Plotly figure with risk dashboard
        """
        logger.info("Creating risk assessment dashboard...")

        try:
            # Load risk data if not provided
            if risk_data is None:
                risk_data = self._load_risk_data()

            if risk_data.empty:
                logger.warning("No risk data available, creating sample")
                risk_data = self._create_sample_risk_data()

            # Create dashboard with 4 subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Risk vs Size & Distance', 'Risk Score Distribution',
                              'Risk Categories', 'Highest Risk Objects'),
                specs=[[{"type": "scatter"}, {"type": "histogram"}],
                       [{"type": "pie"}, {"type": "bar"}]]
            )

            # 1. Scatter plot: Size vs Distance colored by risk
            fig.add_trace(
                go.Scatter(
                    x=risk_data['distance_au'],
                    y=risk_data['h'],
                    mode='markers',
                    marker={
                        "size": np.clip(risk_data['risk_score'] / 5, 5, 30),
                        "color": risk_data['risk_score'],
                        "colorscale": 'Reds',
                        "showscale": True,
                        "colorbar": {"title": "Risk Score", "x": 0.45}
                    },
                    text=risk_data['des'],
                    hovertemplate=(
                        'Object: %{text}<br>Distance: %{x:.6f} AU<br>'
                        'Magnitude: %{y}<br>Risk: %{marker.color}<extra></extra>'
                    ),
                    name='NEO Risk'
                ),
                row=1, col=1
            )

            # 2. Histogram: Risk score distribution
            fig.add_trace(
                go.Histogram(
                    x=risk_data['risk_score'],
                    nbinsx=20,
                    marker_color='orange',
                    opacity=0.7,
                    name='Risk Distribution'
                ),
                row=1, col=2
            )

            # 3. Pie chart: Risk categories
            if 'risk_category' in risk_data.columns:
                risk_counts = risk_data['risk_category'].value_counts()
                fig.add_trace(
                    go.Pie(
                        labels=risk_counts.index,
                        values=risk_counts.values,
                        marker_colors=self.color_palette[:len(risk_counts)],
                        name='Risk Categories'
                    ),
                    row=2, col=1
                )

            # 4. Bar chart: Top 10 riskiest objects
            top_risks = risk_data.nlargest(10, 'risk_score')
            fig.add_trace(
                go.Bar(
                    x=top_risks['des'],
                    y=top_risks['risk_score'],
                    marker_color='red',
                    text=top_risks['risk_score'],
                    textposition='auto',
                    name='Highest Risk'
                ),
                row=2, col=2
            )

            # Update layout
            fig.update_layout(
                title={
                    "text": "Near-Earth Object Risk Assessment Dashboard",
                    "x": 0.5,
                    "font": {"size": 20}
                },
                height=800,
                template='plotly_dark',
                showlegend=False
            )

            # Update axes
            fig.update_xaxes(title_text="Distance (AU)", type='log', row=1, col=1)
            fig.update_yaxes(title_text="Magnitude (H)", row=1, col=1)
            fig.update_xaxes(title_text="Risk Score", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            fig.update_xaxes(title_text="Object Designation", row=2, col=2)
            fig.update_yaxes(title_text="Risk Score", row=2, col=2)

            self.figures['risk_assessment'] = fig
            logger.info("✓ Risk assessment dashboard created")

            return fig

        except (duckdb.Error, ValueError) as e:
            logger.error("Failed to create risk assessment dashboard: %s", e)
            return go.Figure()

    def create_discovery_timeline(self, discovery_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create interactive discovery timeline.

        Args:
            discovery_data: DataFrame with discovery data over time

        Returns:
            Plotly figure with animated timeline
        """
        logger.info("Creating discovery timeline...")

        try:
            # Load discovery data if not provided
            if discovery_data is None:
                discovery_data = self._load_discovery_data()

            if discovery_data.empty:
                logger.warning("No discovery data available, creating sample")
                discovery_data = self._create_sample_discovery_data()

            # Apply size_max from requirements and enhanced color scheme
            timeline_config = self.chart_configs["timeline"]

            # Create animated scatter plot
            fig = px.scatter(
                discovery_data,
                x='distance_au',
                y='magnitude',
                size='risk_score',
                color='risk_category',
                animation_frame='year',
                hover_name='object_name',
                hover_data=['velocity_kms', 'risk_score'],
                title=timeline_config["title"],
                labels={
                    'distance_au': 'Distance (AU)',
                    'magnitude': 'Magnitude (H)',
                    'risk_score': 'Risk Score',
                    'risk_category': 'Risk Category'
                },
                size_max=timeline_config["size_max"],
                color_discrete_sequence=self.color_palette,
                template='plotly_dark'
            )

            # Update layout with enhanced styling from requirements
            timeline_config = self.chart_configs["timeline"]
            fig.update_layout(
                title={
                    "text": timeline_config["title"],
                    "x": 0.5,
                    "font": {"size": 20, "color": "white"}
                },
                xaxis={
                    "type": 'log',
                    "title": 'Distance (AU)',
                    "gridcolor": 'rgba(255,255,255,0.2)',
                    "color": "white"
                },
                yaxis={
                    "title": 'Magnitude (H) - Smaller = Larger Object',
                    "gridcolor": 'rgba(255,255,255,0.2)',
                    "color": "white"
                },
                height=600,
                plot_bgcolor='rgba(0,0,0,0.8)',
                paper_bgcolor='rgba(0,0,0,0.9)',
                font={"color": "white"},
                template='plotly_dark'
            )

            # Update animation settings safely
            try:
                if hasattr(fig.layout, 'updatemenus') and len(fig.layout.updatemenus) > 0:
                    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
                    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
            except (IndexError, KeyError, TypeError):
                logger.warning("Could not update animation settings")

            self.figures['discovery_timeline'] = fig
            logger.info("✓ Discovery timeline created")

            return fig

        except (duckdb.Error, ValueError) as e:
            logger.error("Failed to create discovery timeline: %s", e)
            return go.Figure()

    def create_storage_efficiency_comparison(
        self, efficiency_data: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create storage format efficiency comparison visualization.

        Args:
            efficiency_data: Dictionary with storage efficiency metrics

        Returns:
            Plotly figure comparing storage formats
        """
        logger.info("Creating storage efficiency comparison...")

        try:
            # Load efficiency data if not provided
            if efficiency_data is None:
                efficiency_data = self._create_sample_efficiency_data()

            # Create comparison chart
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('File Size Comparison', 'Compression Ratio', 'Query Performance'),
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
            )

            formats = ['CSV', 'Parquet', 'DuckDB', 'Iceberg']
            file_sizes = [150.5, 45.2, 38.7, 42.1]  # MB
            compression_ratios = [1.0, 3.3, 3.9, 3.6]
            query_speeds = [2.5, 0.8, 0.3, 0.4]  # seconds

            # File size comparison
            fig.add_trace(
                go.Bar(x=formats, y=file_sizes, name='File Size (MB)',
                       marker_color=self.color_palette[:len(formats)],
                       text=[f'{s:.1f} MB' for s in file_sizes],
                       textposition='auto'),
                row=1, col=1
            )

            # Compression ratio
            fig.add_trace(
                go.Bar(x=formats, y=compression_ratios, name='Compression Ratio',
                       marker_color=self.color_palette[:len(formats)],
                       text=[f'{r:.1f}x' for r in compression_ratios],
                       textposition='auto'),
                row=1, col=2
            )

            # Query performance (lower is better)
            fig.add_trace(
                go.Bar(x=formats, y=query_speeds, name='Query Time (s)',
                       marker_color=self.color_palette[:len(formats)],
                       text=[f'{t:.1f}s' for t in query_speeds],
                       textposition='auto'),
                row=1, col=3
            )

            # Update layout
            fig.update_layout(
                title={
                    "text": "Storage Format Efficiency Comparison",
                    "x": 0.5,
                    "font": {"size": 20}
                },
                height=500,
                template='plotly_dark',
                showlegend=False
            )

            # Update axes
            fig.update_yaxes(title_text="Size (MB)", row=1, col=1)
            fig.update_yaxes(title_text="Compression Ratio", row=1, col=2)
            fig.update_yaxes(title_text="Query Time (seconds)", row=1, col=3)

            self.figures['storage_efficiency'] = fig
            logger.info("✓ Storage efficiency comparison created")

            return fig

        except (ValueError, KeyError) as e:
            logger.error("Failed to create storage efficiency comparison: %s", e)
            return go.Figure()

    def generate_presentation_slides(self) -> List[go.Figure]:
        """
        Generate all presentation-ready visualizations.

        Returns:
            List of Plotly figures ready for presentation
        """
        logger.info("Generating presentation slides...")

        try:
            slides = []

            # Generate all visualization types
            performance_chart = self.create_performance_comparison_chart()
            if performance_chart.data:
                slides.append(performance_chart)

            risk_dashboard = self.create_risk_assessment_dashboard()
            if risk_dashboard.data:
                slides.append(risk_dashboard)

            timeline = self.create_discovery_timeline()
            if timeline.data:
                slides.append(timeline)

            efficiency_chart = self.create_storage_efficiency_comparison()
            if efficiency_chart.data:
                slides.append(efficiency_chart)

            logger.info("✓ Generated %d presentation slides", len(slides))
            return slides

        except Exception as e:
            logger.error("Failed to generate presentation slides: %s", e)
            return []

    def save_figures(
        self, output_dir: str = "visualizations", formats: List[str] = None
    ) -> List[str]:
        """
        Save all figures to files.

        Args:
            output_dir: Directory to save figures
            formats: List of formats to save ('png', 'pdf', 'html', 'svg')

        Returns:
            List of saved file paths
        """
        if formats is None:
            formats = ['png', 'html']

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        saved_files = []

        try:
            for fig_name, fig in self.figures.items():
                for fmt in formats:
                    filename = f"{fig_name}.{fmt}"
                    filepath = output_path / filename

                    try:
                        if fmt == 'html':
                            fig.write_html(str(filepath))
                        elif fmt == 'png':
                            fig.write_image(str(filepath), width=1200, height=800)
                        elif fmt == 'pdf':
                            fig.write_image(str(filepath))
                        elif fmt == 'svg':
                            fig.write_image(str(filepath))

                        saved_files.append(str(filepath))
                        logger.debug("✓ Saved %s", filepath)

                    except Exception as e:
                        logger.warning("Failed to save %s: %s", filepath, e)

            logger.info("✓ Saved %d visualization files to %s", len(saved_files), output_path)
            return saved_files

        except Exception as e:
            logger.error("Failed to save figures: %s", e)
            return []

    def _load_risk_data(self) -> pd.DataFrame:
        """Load risk assessment data from database."""
        try:
            query = """
            SELECT
                des,
                fullname,
                h,
                ROUND(dist, 6) as distance_au,
                ROUND(v_rel, 2) as velocity_kms,
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
            LIMIT 100
            """

            return self.conn.execute(query).df()

        except Exception as e:
            logger.warning("Failed to load risk data: %s", e)
            return pd.DataFrame()

    def _load_discovery_data(self) -> pd.DataFrame:
        """Load discovery timeline data from database."""
        try:
            query = """
            SELECT
                des as object_name,
                approach_year as year,
                ROUND(dist, 6) as distance_au,
                h as magnitude,
                ROUND(v_rel, 2) as velocity_kms,
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
            WHERE approach_year IS NOT NULL
                AND approach_year BETWEEN 1900 AND 1967
                AND h IS NOT NULL
                AND dist IS NOT NULL
                AND v_rel IS NOT NULL
            ORDER BY approach_year, risk_score DESC
            LIMIT 500
            """

            return self.conn.execute(query).df()

        except Exception as e:
            logger.warning("Failed to load discovery data: %s", e)
            return pd.DataFrame()

    def _create_sample_performance_data(self) -> Dict:
        """Create sample performance data for demonstration."""
        return {
            'format_comparison': {
                'csv': {
                    'status': 'completed',
                    'simple_count': {'mean': 2.5, 'min': 2.2, 'max': 2.8, 'median': 2.4},
                    'filtered_count': {'mean': 3.1, 'min': 2.9, 'max': 3.4, 'median': 3.0},
                    'aggregation': {'mean': 4.2, 'min': 3.8, 'max': 4.6, 'median': 4.1}
                },
                'parquet': {
                    'status': 'completed',
                    'simple_count': {'mean': 0.8, 'min': 0.7, 'max': 0.9, 'median': 0.8},
                    'filtered_count': {'mean': 1.2, 'min': 1.0, 'max': 1.4, 'median': 1.1},
                    'aggregation': {'mean': 1.5, 'min': 1.3, 'max': 1.7, 'median': 1.4}
                },
                'duckdb_table': {
                    'status': 'completed',
                    'simple_count': {'mean': 0.3, 'min': 0.2, 'max': 0.4, 'median': 0.3},
                    'filtered_count': {'mean': 0.5, 'min': 0.4, 'max': 0.6, 'median': 0.5},
                    'aggregation': {'mean': 0.7, 'min': 0.6, 'max': 0.8, 'median': 0.7}
                }
            }
        }

    def _create_sample_risk_data(self) -> pd.DataFrame:
        """Create sample risk data for demonstration."""
        np.random.seed(42)
        n_objects = 100

        return pd.DataFrame({
            'des': [f'NEO-{i:04d}' for i in range(n_objects)],
            'h': np.random.normal(22, 3, n_objects),
            'distance_au': np.random.lognormal(-2, 1, n_objects),
            'velocity_kms': np.random.normal(20, 8, n_objects),
            'risk_score': np.random.exponential(10, n_objects),
            'risk_category': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'],
                                            n_objects, p=[0.6, 0.25, 0.12, 0.03])
        })

    def _create_sample_discovery_data(self) -> pd.DataFrame:
        """Create sample discovery timeline data."""
        np.random.seed(42)
        years = list(range(2000, 2025))
        data = []

        for year in years:
            n_discoveries = np.random.poisson(20)
            for i in range(n_discoveries):
                data.append({
                    'object_name': f'{year}-{i:03d}',
                    'year': year,
                    'distance_au': np.random.lognormal(-2, 1),
                    'magnitude': np.random.normal(22, 3),
                    'velocity_kms': np.random.normal(20, 8),
                    'risk_score': np.random.exponential(10),
                    'risk_category': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'],
                                                    p=[0.6, 0.25, 0.12, 0.03])
                })

        return pd.DataFrame(data)

    def _parse_performance_markdown(self, md_file: Path) -> Dict:
        """Parse performance data from markdown report."""
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the executive summary table
            format_comparison = {}
            lines = content.split('\n')

            # Look for the summary table
            in_summary_table = False
            for line in lines:
                if '| Format | Avg Query Time' in line:
                    in_summary_table = True
                    continue
                elif in_summary_table and line.startswith('|') and '---' not in line:
                    # Parse table rows like: | iceberg | 0.0019 | 4 | #1 |
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 3:
                        format_name = parts[0].lower()
                        avg_time = float(parts[1])
                        queries_tested = int(parts[2])

                        # Create format entry with dummy query results
                        format_comparison[format_name] = {
                            'status': 'completed',
                            'simple_count': {'mean': avg_time, 'min': avg_time * 0.9, 'max': avg_time * 1.1, 'median': avg_time},
                            'filtered_count': {'mean': avg_time * 1.2, 'min': avg_time * 1.1, 'max': avg_time * 1.3, 'median': avg_time * 1.2},
                            'aggregation': {'mean': avg_time * 1.5, 'min': avg_time * 1.4, 'max': avg_time * 1.6, 'median': avg_time * 1.5}
                        }
                elif in_summary_table and not line.startswith('|'):
                    break

            # Also parse detailed sections for more accurate data
            self._parse_detailed_sections(content, format_comparison)

            return {'format_comparison': format_comparison}

        except Exception as e:
            logger.warning("Failed to parse markdown file: %s", e)
            return self._create_sample_performance_data()

    def _parse_detailed_sections(self, content: str, format_comparison: Dict) -> None:
        """Parse detailed query results from markdown sections."""
        try:
            lines = content.split('\n')
            current_format = None
            current_query = None

            for line in lines:
                line = line.strip()

                # Detect format sections
                if line.startswith('### ') and 'Format' in line:
                    format_name = line.replace('### ', '').replace(' Format', '').lower()
                    current_format = format_name

                # Detect query subsections
                elif line.startswith('**') and line.endswith(':**'):
                    query_name = line.replace('**', '').replace(':', '').lower().replace(' ', '_')
                    current_query = query_name

                # Parse timing data
                elif current_format and current_query and line.startswith('- Mean:'):
                    try:
                        mean_time = float(line.split(':')[1].replace('s', '').strip())
                        if current_format in format_comparison:
                            if current_query not in format_comparison[current_format]:
                                format_comparison[current_format][current_query] = {}
                            format_comparison[current_format][current_query]['mean'] = mean_time
                    except (ValueError, IndexError):
                        continue

                elif current_format and current_query and line.startswith('- Median:'):
                    try:
                        median_time = float(line.split(':')[1].replace('s', '').strip())
                        if current_format in format_comparison and current_query in format_comparison[current_format]:
                            format_comparison[current_format][current_query]['median'] = median_time
                    except (ValueError, IndexError):
                        continue

                elif current_format and current_query and line.startswith('- Min:'):
                    try:
                        min_time = float(line.split(':')[1].replace('s', '').strip())
                        if current_format in format_comparison and current_query in format_comparison[current_format]:
                            format_comparison[current_format][current_query]['min'] = min_time
                    except (ValueError, IndexError):
                        continue

                elif current_format and current_query and line.startswith('- Max:'):
                    try:
                        max_time = float(line.split(':')[1].replace('s', '').strip())
                        if current_format in format_comparison and current_query in format_comparison[current_format]:
                            format_comparison[current_format][current_query]['max'] = max_time
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            logger.warning("Failed to parse detailed sections: %s", e)

    def _create_sample_efficiency_data(self) -> Dict:
        """Create sample storage efficiency data."""
        return {
            'file_sizes': {
                'csv': {'size_mb': 150.5},
                'parquet': {'size_mb': 45.2},
                'duckdb': {'size_mb': 38.7},
                'iceberg': {'size_mb': 42.1}
            },
            'compression_ratios': {
                'parquet': 3.3,
                'duckdb': 3.9,
                'iceberg': 3.6
            }
        }

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("✓ Visualizer database connection closed")


def main():
    """Main execution function for visualization generation."""
    logger.info("=" * 60)
    logger.info("SPACE DATA VISUALIZATION")
    logger.info("=" * 60)

    visualizer = None

    try:
        # Initialize visualizer
        visualizer = SpaceDataVisualizer()

        # Generate all visualizations
        logger.info("\nGenerating visualizations...")

        # Generate all visualizations
        visualizer.create_performance_comparison_chart()
        visualizer.create_risk_assessment_dashboard()
        visualizer.create_discovery_timeline()
        visualizer.create_storage_efficiency_comparison()

        # Save all figures
        saved_files = visualizer.save_figures(formats=['html', 'png'])

        # Summary
        logger.info("\n%s", "="*60)
        logger.info("VISUALIZATION SUMMARY")
        logger.info("="*60)
        logger.info("Figures created: %d", len(visualizer.figures))

        for fig_name in visualizer.figures:
            logger.info("  ✓ %s", fig_name.replace('_', ' ').title())

        logger.info("Files saved: %d", len(saved_files))
        for file_path in saved_files[:5]:  # Show first 5
            logger.info("  ✓ %s", file_path)

        if len(saved_files) > 5:
            logger.info("  ... and %d more files", len(saved_files) - 5)

        logger.info("\n🎉 Data visualization completed successfully!")
        logger.info("Check the 'visualizations' directory for all charts")

        return True

    except Exception as e:
        logger.error("❌ Data visualization failed: %s", e)
        return False
    finally:
        if visualizer:
            visualizer.close()


if __name__ == "__main__":
    import sys
    SUCCESS = main()
    sys.exit(0 if SUCCESS else 1)
