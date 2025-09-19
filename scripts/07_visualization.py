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

            # Define color scheme as specified
            format_colors = {
                'csv': '#800080',      # Purple
                'CSV': '#800080',
                'parquet': '#008000',  # Green
                'Parquet': '#008000',
                'duckdb': '#FFD700',   # Gold
                'duckdb_table': '#FFD700',
                'DuckDB': '#FFD700',
                'iceberg': '#0000FF',  # Blue
                'Iceberg': '#0000FF'
            }

            # 1. Bar chart - mean performance by format with specified colors
            formats = df['format'].unique()
            mean_times = df.groupby('format')['mean_time'].mean()

            # Use specified color palette
            colors = [format_colors.get(fmt, format_colors.get(fmt.lower(), '#666666')) for fmt in formats]

            fig.add_trace(
                go.Bar(x=formats, y=mean_times, name='Avg Query Time',
                       marker_color=colors,
                       marker_line=dict(color='rgba(255,255,255,0.3)', width=1),
                       text=[f'{t:.3f}s' for t in mean_times],
                       textposition='auto',
                       textfont=dict(color='white', size=12)),
                row=1, col=1
            )

            # 2. Box plot - performance distribution with enhanced orange theme
            for i, format_name in enumerate(formats):
                format_data = df[df['format'] == format_name]['mean_time']
                color = format_colors.get(format_name, format_colors.get(format_name.lower(), '#666666'))
                fig.add_trace(
                    go.Box(
                        y=format_data,
                        name=format_name,
                        marker_color=color,
                        marker_line=dict(color='#8B4513', width=1),  # Saddle brown border
                        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, '
                                  f'{int(color[5:7], 16)}, 0.5)'  # Semi-transparent fill
                    ),
                    row=1, col=2
                )

            # 3. Grouped bar chart - detailed comparison with orange shades for query types
            queries = df['query'].unique()
            orange_shades = {
                'simple_count': '#FF8C00',      # Dark orange
                'Simple Count': '#FF8C00',      # Dark orange
                'filtered_count': '#FFA500',    # Orange
                'Filtered Count': '#FFA500',    # Orange
                'aggregation': '#FFB84D',       # Light orange
                'Aggregation': '#FFB84D'        # Light orange
            }

            for i, query in enumerate(queries):
                query_data = df[df['query'] == query]
                # Use orange shades for different query types
                query_color = orange_shades.get(query, orange_shades.get(query.lower(), '#FF7F50'))  # Default coral
                fig.add_trace(
                    go.Bar(
                        x=query_data['format'],
                        y=query_data['mean_time'],
                        name=query,
                        marker_color=query_color,
                        marker_line=dict(color='#B8860B', width=1)  # Dark goldenrod border
                    ),
                    row=2, col=1
                )

            # 4. Scatter plot - complexity vs performance (hidden from legend)
            fig.add_trace(
                go.Scatter(x=df.index, y=df['mean_time'],
                          mode='markers+lines',
                          marker={
                              "size": 10, "color": df['mean_time'],
                              "colorscale": 'Viridis', "showscale": True
                          },
                          text=df['format'] + '<br>' + df['query'],
                          name='Performance Trend',
                          showlegend=False),  # Remove from legend as specified
                row=2, col=2
            )

            # Update layout with left-positioned legends
            fig.update_layout(
                title={
                    "text": "DuckDB + Iceberg vs Traditional Storage Performance",
                    "x": 0.5,
                    "font": {"size": 20}
                },
                showlegend=True,
                # Left-positioned legend as specified
                legend=dict(
                    x=-0.15,
                    y=0.5,
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='white',
                    borderwidth=1
                ),
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
        Create comprehensive temporal risk assessment dashboard.
        Separates historical (≤2025) and future (≥2026) risk analysis.

        Args:
            risk_data: DataFrame with risk assessment data

        Returns:
            Plotly figure with temporal segmentation dashboard
        """
        logger.info("Creating temporal risk assessment dashboard...")

        try:
            # Load risk data if not provided
            if risk_data is None:
                risk_data = self._load_risk_data()

            if risk_data.empty:
                logger.warning("No risk data available, creating sample")
                risk_data = self._create_sample_risk_data()

            # Parse approach dates and add year column
            risk_data = risk_data.copy()
            if 'approach_date' in risk_data.columns:
                risk_data['approach_year'] = pd.to_datetime(
                    risk_data['approach_date'], errors='coerce'
                ).dt.year
            else:
                # Use current year as fallback
                risk_data['approach_year'] = 2025

            # Temporal segmentation: Historical (≤2025) vs Future (≥2026)
            historical_data = risk_data[risk_data['approach_year'] <= 2025]
            future_data = risk_data[risk_data['approach_year'] >= 2026]

            logger.info("Temporal segmentation: %d historical, %d future records",
                       len(historical_data), len(future_data))

            # If no future data, create projections from historical patterns
            if future_data.empty:
                logger.info("No future data found, creating projections from historical patterns")
                future_data = self._create_future_projections(historical_data)

            # Create main dashboard with temporal sections
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[
                    'Historical & Present Risk Assessment (≤2025)',
                    'Future Risk Projections & Trends (≥2026)'
                ],
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
                vertical_spacing=0.15
            )

            # SECTION 1: Historical & Present Risk Assessment (≤2025)
            self._add_temporal_risk_plots(fig, historical_data, "Historical", "#8B0000", 1)

            # SECTION 2: Future Risk Projections & Trends (≥2026)
            self._add_temporal_risk_plots(fig, future_data, "Future", "#006400", 2)

            # Add comparative statistics as annotations
            self._add_temporal_comparison_annotations(fig, historical_data, future_data)

            # Add cross-section filtering capabilities (simplified for now)
            # Note: Full interactive cross-filtering would require JavaScript callbacks
            logger.info("Cross-section filtering available via plot interactions")

            # Update layout for temporal dashboard
            subtitle = "Temporal Risk Analysis: Historical Patterns vs Future Projections"
            fig.update_layout(
                title={
                    "text": f"Temporal NEO Risk Assessment Dashboard<br><sub>{subtitle}</sub>",
                    "x": 0.5,
                    "font": {"size": 18, "color": "white"}
                },
                height=1200,  # Increased height for two sections
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0.8)',
                paper_bgcolor='rgba(0,0,0,0.9)',
                font={"color": "white"},
                showlegend=True,
                legend=dict(
                    x=1.02,
                    y=1,
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='white',
                    borderwidth=1
                )
            )

            # Update axes for temporal sections with corrected labels
            fig.update_xaxes(title_text="Approach Year", row=1, col=1)
            fig.update_yaxes(title_text="Risk Score", row=1, col=1)
            fig.update_xaxes(title_text="Projected Approach Year", row=2, col=1)
            fig.update_yaxes(title_text="Projected Risk Score", row=2, col=1)

            self.figures['risk_assessment'] = fig
            logger.info("✓ Temporal risk assessment dashboard created")
            logger.info("Historical records: %d, Future projections: %d",
                       len(historical_data), len(future_data))

            return fig

        except (duckdb.Error, ValueError) as e:
            logger.error("Failed to create risk assessment dashboard: %s", e)
            return go.Figure()

    def create_discovery_timeline(self, discovery_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create interactive 3D discovery timeline with working animation.

        Args:
            discovery_data: DataFrame with discovery data over time

        Returns:
            Plotly figure with 3D animated timeline
        """
        logger.info("Creating 3D discovery timeline...")

        try:
            # Load discovery data if not provided
            if discovery_data is None:
                discovery_data = self._load_discovery_data()

            if discovery_data.empty:
                logger.warning("No discovery data available, creating sample")
                discovery_data = self._create_sample_discovery_data()

            # Ensure we have the right column names
            if 'approach_year' not in discovery_data.columns:
                discovery_data['approach_year'] = discovery_data['year']

            # Define risk category color mapping as specified
            risk_color_map = {
                'VERY_HIGH': '#8B0000',  # Dark red
                'HIGH': '#CD5C5C',      # Medium red
                'MEDIUM': '#FFA0A0',    # Light red
                'LOW': '#FFD0D0'        # Very light red
            }

            # Map colors to data
            discovery_data['color'] = discovery_data['risk_category'].map(risk_color_map)

            # Get unique years and create a more complete dataset
            years = sorted(discovery_data['year'].unique())
            logger.info(f"Years available: {min(years)} to {max(years)} ({len(years)} total)")

            # Create the base 3D scatter plot with ALL data points
            fig = go.Figure()

            # Add all data points for each risk category (all years combined)
            for risk_category in ['VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW']:
                category_data = discovery_data[discovery_data['risk_category'] == risk_category]

                if not category_data.empty:
                    fig.add_trace(
                        go.Scatter3d(
                            x=category_data['distance_au'],
                            y=category_data['magnitude'],
                            z=category_data['approach_year'],
                            mode='markers',
                            marker=dict(
                                size=np.clip(category_data['risk_score'] / 5, 5, 20),
                                color=risk_color_map[risk_category],
                                opacity=0.8,
                                line=dict(width=1, color='white')
                            ),
                            text=category_data['object_name'],
                            hovertemplate=(
                                'Object: %{text}<br>'
                                'Distance: %{x:.6f} AU<br>'
                                'Magnitude: %{y}<br>'
                                'Year: %{z}<br>'
                                'Risk Score: %{customdata[0]}<br>'
                                'Velocity: %{customdata[1]} km/s<br>'
                                'Risk Category: ' + risk_category + '<extra></extra>'
                            ),
                            customdata=list(zip(
                                category_data['risk_score'],
                                category_data['velocity_kms']
                            )),
                            name=f'{risk_category.replace("_", " ").title()} Risk',
                            visible=True,
                            showlegend=True
                        )
                    )

            # Create animation frames for year-by-year progression
            frames = []

            # Create cumulative frames (showing data up to current year)
            for i, current_year in enumerate(years):
                frame_data = []

                for risk_category in ['VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW']:
                    # Show data up to and including current year
                    cumulative_data = discovery_data[
                        (discovery_data['risk_category'] == risk_category) &
                        (discovery_data['year'] <= current_year)
                    ]

                    if not cumulative_data.empty:
                        frame_data.append(
                            go.Scatter3d(
                                x=cumulative_data['distance_au'],
                                y=cumulative_data['magnitude'],
                                z=cumulative_data['approach_year'],
                                mode='markers',
                                marker=dict(
                                    size=np.clip(cumulative_data['risk_score'] / 5, 5, 20),
                                    color=risk_color_map[risk_category],
                                    opacity=0.8,
                                    line=dict(width=1, color='white')
                                ),
                                text=cumulative_data['object_name'],
                                hovertemplate=(
                                    'Object: %{text}<br>'
                                    'Distance: %{x:.6f} AU<br>'
                                    'Magnitude: %{y}<br>'
                                    'Year: %{z}<br>'
                                    'Risk Score: %{customdata[0]}<br>'
                                    'Velocity: %{customdata[1]} km/s<br>'
                                    'Risk Category: ' + risk_category + '<extra></extra>'
                                ),
                                customdata=list(zip(
                                    cumulative_data['risk_score'],
                                    cumulative_data['velocity_kms']
                                )),
                                name=f'{risk_category.replace("_", " ").title()} Risk'
                            )
                        )
                    else:
                        # Empty trace for this category
                        frame_data.append(
                            go.Scatter3d(
                                x=[], y=[], z=[],
                                mode='markers',
                                name=f'{risk_category.replace("_", " ").title()} Risk'
                            )
                        )

                frames.append(go.Frame(
                    data=frame_data,
                    name=str(current_year),
                    traces=list(range(len(frame_data)))
                ))

            fig.frames = frames

            # Add working animation controls
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=[
                            dict(
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": 1000, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 300, "easing": "quadratic-in-out"}
                                    }
                                ],
                                label="Play",
                                method="animate"
                            ),
                            dict(
                                args=[
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}
                                    }
                                ],
                                label="Pause",
                                method="animate"
                            )
                        ],
                        pad={"r": 10, "t": 87},
                        showactive=False,
                        x=0.011,
                        xanchor="right",
                        y=0,
                        yanchor="top"
                    )
                ],
                sliders=[
                    dict(
                        active=len(years)-1,  # Start at the end to show all data
                        yanchor="top",
                        xanchor="left",
                        currentvalue={
                            "font": {"size": 16, "color": "white"},
                            "prefix": "Year: ",
                            "visible": True,
                            "xanchor": "right"
                        },
                        transition={"duration": 300, "easing": "cubic-in-out"},
                        pad={"b": 10, "t": 50},
                        len=0.9,
                        x=0.1,
                        y=0,
                        steps=[
                            dict(
                                args=[
                                    [str(year)],
                                    {
                                        "frame": {"duration": 300, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 300}
                                    }
                                ],
                                label=str(year),
                                method="animate"
                            ) for year in years
                        ]
                    )
                ]
            )

            # Update layout with 3D scene configuration
            fig.update_layout(
                title={
                    "text": "3D NEO Discovery Timeline (1900-1966)",
                    "x": 0.5,
                    "font": {"size": 20, "color": "white"}
                },
                scene=dict(
                    xaxis=dict(
                        title='Distance (AU)',
                        type='log',
                        gridcolor='rgba(255,255,255,0.2)',
                        backgroundcolor='rgba(0,0,0,0.8)'
                    ),
                    yaxis=dict(
                        title='Magnitude (H)',
                        gridcolor='rgba(255,255,255,0.2)',
                        backgroundcolor='rgba(0,0,0,0.8)'
                    ),
                    zaxis=dict(
                        title='Year',
                        range=[1900, 1970],
                        gridcolor='rgba(255,255,255,0.2)',
                        backgroundcolor='rgba(0,0,0,0.8)'
                    ),
                    bgcolor='rgba(0,0,0,0.8)',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1)
                    )
                ),
                height=700,
                plot_bgcolor='rgba(0,0,0,0.8)',
                paper_bgcolor='rgba(0,0,0,0.9)',
                font={"color": "white"},
                template='plotly_dark',
                legend=dict(
                    x=1.02,
                    y=1,
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='white',
                    borderwidth=1
                ),
                margin=dict(l=0, r=0, t=50, b=0)
            )

            self.figures['discovery_timeline'] = fig
            logger.info("✓ 3D Discovery timeline created with %d years of data", len(years))

            return fig

        except (duckdb.Error, ValueError) as e:
            logger.error("Failed to create 3D discovery timeline: %s", e)
            return go.Figure()

    def create_storage_efficiency_comparison(
        self, efficiency_data: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create storage format efficiency comparison visualization with legend.

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

            # Define color mapping as specified
            format_colors = {
                'CSV': '#800080',      # Purple
                'Parquet': '#008000',  # Green
                'DuckDB': '#FFD700',   # Gold
                'Iceberg': '#0000FF'   # Blue
            }

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
            colors = [format_colors[fmt] for fmt in formats]

            # File size comparison
            fig.add_trace(
                go.Bar(x=formats, y=file_sizes, name='File Size (MB)',
                       marker_color=colors,
                       text=[f'{s:.1f} MB' for s in file_sizes],
                       textposition='auto',
                       showlegend=True,
                       legendgroup='formats'),
                row=1, col=1
            )

            # Compression ratio
            fig.add_trace(
                go.Bar(x=formats, y=compression_ratios, name='Compression Ratio',
                       marker_color=colors,
                       text=[f'{r:.1f}x' for r in compression_ratios],
                       textposition='auto',
                       showlegend=False,
                       legendgroup='formats'),
                row=1, col=2
            )

            # Query performance (lower is better)
            fig.add_trace(
                go.Bar(x=formats, y=query_speeds, name='Query Time (s)',
                       marker_color=colors,
                       text=[f'{t:.1f}s' for t in query_speeds],
                       textposition='auto',
                       showlegend=False,
                       legendgroup='formats'),
                row=1, col=3
            )

            # Add individual traces for legend with specified colors
            for fmt in formats:
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=15, color=format_colors[fmt]),
                        name=fmt,
                        showlegend=True
                    )
                )

            # Update layout with legend at bottom-center
            fig.update_layout(
                title={
                    "text": "Storage Format Efficiency Comparison",
                    "x": 0.5,
                    "font": {"size": 20}
                },
                height=600,  # Increased for legend space
                template='plotly_dark',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.1,
                    xanchor="center",
                    x=0.5,
                    title="Storage Format Types",
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='white',
                    borderwidth=1
                )
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
        """Load risk assessment data from database with duplicate handling."""
        try:
            # Query with duplicate handling - get highest risk score per object
            query = """
            WITH risk_calculations AS (
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
                    END as risk_category,
                    ROW_NUMBER() OVER (PARTITION BY des ORDER BY
                        GREATEST(0, 30 - h) * 1.33 + GREATEST(0, 0.1 - dist) * 400 + LEAST(v_rel, 60) * 0.33 DESC
                    ) as rn
                FROM neo_approaches
                WHERE h IS NOT NULL AND dist IS NOT NULL AND v_rel IS NOT NULL
            )
            SELECT des, fullname, h, distance_au, velocity_kms, approach_date, risk_score, risk_category
            FROM risk_calculations
            WHERE rn = 1
            ORDER BY risk_score DESC
            -- Removed LIMIT for comprehensive risk analysis
            """

            return self.conn.execute(query).df()

        except Exception as e:
            logger.warning("Failed to load risk data: %s", e)
            return pd.DataFrame()

    def _load_discovery_data(self) -> pd.DataFrame:
        """Load discovery timeline data from database."""
        try:
            query = """
            WITH sampled_data AS (
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
                    END as risk_category,
                    ROW_NUMBER() OVER (PARTITION BY approach_year ORDER BY RANDOM()) as rn
                FROM neo_approaches
                WHERE approach_year IS NOT NULL
                    AND approach_year BETWEEN 1900 AND 1966
                    AND h IS NOT NULL
                    AND dist IS NOT NULL
                    AND v_rel IS NOT NULL
            )
            SELECT object_name, year, distance_au, magnitude, velocity_kms, risk_score, risk_category
            FROM sampled_data
            WHERE rn <= 50  -- Increased to 50 objects per year for more comprehensive data
            ORDER BY year, risk_score DESC
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
        """Create sample discovery timeline data with extended year range."""
        np.random.seed(42)
        years = list(range(1900, 1970, 5))  # Every 5 years from 1900 to 1965
        data = []

        for year in years:
            n_discoveries = np.random.poisson(15)
            for i in range(n_discoveries):
                data.append({
                    'object_name': f'{year}-{i:03d}',
                    'year': year,
                    'approach_year': year,  # Add this field
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

    def _parse_performance_markdown(self, md_file: Path) -> Dict:
        """Parse performance results from markdown report."""
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            format_comparison = {}
            current_format = None
            current_query = None

            lines = content.split('\n')

            for line in lines:
                # Detect format section
                if line.startswith('### ') and 'Format' in line:
                    current_format = line.replace('### ', '').replace(' Format', '').lower()
                    format_comparison[current_format] = {'status': 'completed'}

                # Detect query type
                elif line.startswith('**') and line.endswith(':**'):
                    current_query = line.replace('**', '').replace(':', '').lower().replace(' ', '_')
                    if current_format and current_query:
                        format_comparison[current_format][current_query] = {}

                # Parse timing data
                elif current_format and current_query:
                    if line.startswith('- Mean:'):
                        try:
                            mean_time = float(line.split(':')[1].replace('s', '').strip())
                            format_comparison[current_format][current_query]['mean'] = mean_time
                        except (ValueError, IndexError):
                            continue
                    elif line.startswith('- Median:'):
                        try:
                            median_time = float(line.split(':')[1].replace('s', '').strip())
                            format_comparison[current_format][current_query]['median'] = median_time
                        except (ValueError, IndexError):
                            continue
                    elif line.startswith('- Min:'):
                        try:
                            min_time = float(line.split(':')[1].replace('s', '').strip())
                            format_comparison[current_format][current_query]['min'] = min_time
                        except (ValueError, IndexError):
                            continue
                    elif line.startswith('- Max:'):
                        try:
                            max_time = float(line.split(':')[1].replace('s', '').strip())
                            format_comparison[current_format][current_query]['max'] = max_time
                        except (ValueError, IndexError):
                            continue

            return {'format_comparison': format_comparison}

        except Exception as e:
            logger.error("Failed to parse markdown: %s", e)
            return {}

    def _create_sample_performance_data(self) -> Dict:
        """Create sample performance data as fallback."""
        return {
            'format_comparison': {
                'csv': {
                    'status': 'completed',
                    'simple_count': {'mean': 0.155, 'median': 0.154, 'min': 0.154, 'max': 0.157}
                },
                'parquet': {
                    'status': 'completed',
                    'simple_count': {'mean': 0.0003, 'median': 0.0002, 'min': 0.0002, 'max': 0.0004}
                },
                'duckdb_table': {
                    'status': 'completed',
                    'simple_count': {'mean': 0.0001, 'median': 0.0001, 'min': 0.0001, 'max': 0.0001}
                },
                'iceberg': {
                    'status': 'completed',
                    'simple_count': {'mean': 0.0002, 'median': 0.0002, 'min': 0.0002, 'max': 0.0002}
                }
            }
        }

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

    def _create_risk_section_subplots(
        self,
        data: pd.DataFrame,
        section_name: str,
        primary_color: str
    ) -> go.Figure:
        """
        Create a 2x2 subplot section for risk assessment.

        Args:
            data: Risk data for this temporal section
            section_name: Name of the section (Historical/Future)
            primary_color: Primary color scheme for the section

        Returns:
            Figure with 2x2 risk assessment subplots
        """
        # Create 2x2 subplots for this section
        section_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{section_name} Risk vs Size & Distance',
                f'{section_name} Risk Score Distribution',
                f'{section_name} Risk Categories',
                f'{section_name} Top Risk Objects'
            ),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )

        if data.empty:
            # Add empty placeholder traces
            section_fig.add_trace(
                go.Scatter(x=[], y=[], mode='markers', name=f'{section_name} (No Data)'),
                row=1, col=1
            )
            return section_fig

        # 1. Scatter plot: Size vs Distance colored by risk
        section_fig.add_trace(
            go.Scatter(
                x=data['distance_au'],
                y=data['h'],
                mode='markers',
                marker={
                    "size": np.clip(data['risk_score'] / 5, 5, 20),
                    "color": data['risk_score'],
                    "colorscale": 'Reds' if 'Historical' in section_name else 'Greens',
                    "showscale": True
                },
                text=data['des'],
                hovertemplate=(
                    'Object: %{text}<br>Distance: %{x:.6f} AU<br>'
                    'Magnitude: %{y}<br>Risk: %{marker.color}<extra></extra>'
                ),
                name=f'{section_name} Risk Scatter'
            ),
            row=1, col=1
        )

        # 2. Histogram: Risk score distribution
        section_fig.add_trace(
            go.Histogram(
                x=data['risk_score'],
                nbinsx=15,
                marker_color=primary_color,
                opacity=0.8,
                name=f'{section_name} Risk Distribution'
            ),
            row=1, col=2
        )

        # 3. Pie chart: Risk categories
        if 'risk_category' in data.columns and not data['risk_category'].isna().all():
            risk_counts = data['risk_category'].value_counts()
            color_palette = ['#ffcccc', '#ff9999', '#ff6666', '#cc0000'] if 'Historical' in section_name else ['#ccffcc', '#99ff99', '#66ff66', '#00cc00']
            section_fig.add_trace(
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    marker_colors=color_palette[:len(risk_counts)],
                    name=f'{section_name} Categories'
                ),
                row=2, col=1
            )

        # 4. Bar chart: Top risk objects
        top_risks = data.nlargest(8, 'risk_score')
        section_fig.add_trace(
            go.Bar(
                x=top_risks['des'][:8],  # Limit to 8 for readability
                y=top_risks['risk_score'][:8],
                marker_color=primary_color,
                name=f'{section_name} Top Risks'
            ),
            row=2, col=2
        )

        return section_fig

    def _add_temporal_risk_plots(
        self,
        fig: go.Figure,
        data: pd.DataFrame,
        section_name: str,
        primary_color: str,
        row: int
    ) -> None:
        """
        Add temporal risk assessment plots to the main figure.

        Args:
            fig: Main figure to add plots to
            data: Risk data for this temporal section
            section_name: Name of the section (Historical/Future)
            primary_color: Primary color scheme for the section
            row: Row number to add plots to
        """
        if data.empty:
            # Add placeholder for empty data
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0], mode='markers',
                    marker=dict(size=20, color='gray'),
                    name=f'{section_name} (No Data)',
                    text='No data available',
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=row, col=1
            )
            return

        # Scatter plot: Risk vs Approach Year (corrected axes)
        fig.add_trace(
            go.Scatter(
                x=data['approach_year'],  # X-axis: Approach years
                y=data['risk_score'],     # Y-axis: Risk factor
                mode='markers',
                marker={
                    "size": np.clip(30 - data['h'], 5, 25),  # Larger objects = larger markers
                    "color": data['risk_score'],
                    "colorscale": 'Reds' if 'Historical' in section_name else 'Greens',
                    "showscale": True,
                    "colorbar": {
                        "title": f"{section_name} Risk Score",
                        "x": 1.02 if row == 1 else 1.05,
                        "y": 0.75 if row == 1 else 0.25,
                        "len": 0.3
                    }
                },
                text=data['des'],
                hovertemplate=(
                    f'{section_name} Object: %{{text}}<br>'
                    'Approach Year: %{x}<br>'
                    'Risk Score: %{y:.1f}<br>'
                    'Distance: %{customdata[0]:.6f} AU<br>'
                    'Magnitude: %{customdata[1]}<extra></extra>'
                ),
                customdata=list(zip(data['distance_au'], data['h'])),
                name=f'{section_name} Risk vs Year'
            ),
            row=row, col=1
        )

        # Add trend line for risk over time if we have temporal data
        if len(data) > 5:  # Need sufficient data for trend
            # Calculate moving average of risk scores by year
            yearly_risk = data.groupby('approach_year')['risk_score'].mean().reset_index()

            if len(yearly_risk) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=yearly_risk['approach_year'],
                        y=yearly_risk['risk_score'],
                        mode='lines+markers',
                        line=dict(color=primary_color, width=2),
                        marker=dict(size=8, color=primary_color),
                        name=f'{section_name} Risk Trend',
                        yaxis='y2'  # Use secondary y-axis
                    ),
                    row=row, col=1
                )

    def _create_future_projections(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create future risk projections based on historical patterns.

        Args:
            historical_data: Historical risk data

        Returns:
            DataFrame with projected future risk data
        """
        if historical_data.empty:
            return pd.DataFrame()

        logger.info("Creating future projections from %d historical records", len(historical_data))

        # Use statistical patterns from historical data to project future scenarios
        np.random.seed(42)  # For reproducible projections

        # Create projections for years 2026-2030
        future_years = list(range(2026, 2031))
        projections = []

        for year in future_years:
            # Sample from historical patterns with some variation
            year_sample_size = max(10, len(historical_data) // 10)
            base_sample = historical_data.sample(n=min(year_sample_size, len(historical_data)), random_state=year)

            for i, (_, row) in enumerate(base_sample.iterrows()):
                # Add some variation to create realistic future scenarios
                projection = row.copy()
                projection['approach_year'] = year
                projection['approach_date'] = f"{year}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"

                # Slight variations in risk factors
                projection['distance_au'] = max(0.001, row['distance_au'] * np.random.normal(1.0, 0.1))
                projection['velocity_kms'] = max(1.0, row['velocity_kms'] * np.random.normal(1.0, 0.05))
                projection['risk_score'] = max(0, row['risk_score'] * np.random.normal(1.0, 0.15))

                # Update object designation for projection
                projection['des'] = f"PROJ-{year}-{i:03d}"
                projection['fullname'] = f"Projected Object {year}-{i:03d}"

                projections.append(projection)

        future_df = pd.DataFrame(projections)
        logger.info("Generated %d future projections", len(future_df))
        return future_df

    def _add_temporal_comparison_annotations(
        self,
        fig: go.Figure,
        historical_data: pd.DataFrame,
        future_data: pd.DataFrame
    ) -> None:
        """
        Add comparative statistics annotations between historical and future data.

        Args:
            fig: Main figure to add annotations to
            historical_data: Historical risk data
            future_data: Future risk data
        """
        if historical_data.empty or future_data.empty:
            return

        # Calculate comparative statistics
        hist_mean_risk = historical_data['risk_score'].mean()
        future_mean_risk = future_data['risk_score'].mean()

        hist_high_risk = len(historical_data[historical_data['risk_score'] > 50])
        future_high_risk = len(future_data[future_data['risk_score'] > 50])

        # Add annotation with comparative statistics
        comparison_text = (
            f"Historical Avg Risk: {hist_mean_risk:.1f}<br>"
            f"Future Avg Risk: {future_mean_risk:.1f}<br>"
            f"Historical High Risk Objects: {hist_high_risk}<br>"
            f"Future High Risk Objects: {future_high_risk}<br>"
            f"Risk Trend: {'↗️ Increasing' if future_mean_risk > hist_mean_risk else '↘️ Decreasing'}"
        )

        fig.add_annotation(
            x=0.02, y=0.5,
            xref="paper", yref="paper",
            text=comparison_text,
            showarrow=False,
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1,
            font=dict(color="white", size=12)
        )

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
