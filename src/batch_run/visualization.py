"""
Helper functions to create discrepancy timeline visualizations.
Designed to be imported into Quarto notebooks - no intermediate files saved.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def check_cfr_consistency(df, year_col='Year', week_col='WeekNumber'):
    """
    Calculate CFR consistency for each record.

    Returns dataframe with 'cfr_error' column showing how far
    calculated CFR (Deaths/TotalCases*100) deviates from stated CFR.

    Args:
        df: Dataframe with TotalCases, Deaths, CFR columns
        year_col: Name of year column
        week_col: Name of week column

    Returns:
        DataFrame with added cfr_error column
    """
    df = df.copy()

    # Convert to numeric, handling strings and nulls
    df['TotalCases_num'] = pd.to_numeric(df['TotalCases'], errors='coerce')
    df['Deaths_num'] = pd.to_numeric(df['Deaths'], errors='coerce')
    df['CFR_num'] = pd.to_numeric(df['CFR'], errors='coerce')

    # Only calculate where we have required fields
    mask = (df['TotalCases_num'] > 0) & (df['Deaths_num'].notna()) & (df['CFR_num'].notna())

    df['calculated_cfr'] = np.nan
    df.loc[mask, 'calculated_cfr'] = (df.loc[mask, 'Deaths_num'] / df.loc[mask, 'TotalCases_num']) * 100

    df['cfr_error'] = np.nan
    df.loc[mask, 'cfr_error'] = abs(df.loc[mask, 'calculated_cfr'] - df.loc[mask, 'CFR_num'])

    # Clean up temporary columns
    df.drop(['TotalCases_num', 'Deaths_num', 'CFR_num'], axis=1, inplace=True)

    return df


def create_timeline_plot(batch_df, ruleb_df, country, event,
                         week_col='WeekNumber', year_col='Year',
                         highlight_cfr_winner=True, height=400):
    """
    Create a timeline comparison plot for a specific country-event pair.

    Green outline highlights the system with better CFR consistency at each point.

    Args:
        batch_df: LLM batch extraction dataframe
        ruleb_df: Rule-based baseline dataframe
        country: Country name
        event: Event name
        week_col: Name of week column (default: 'WeekNumber')
        year_col: Name of year column (default: 'Year')
        highlight_cfr_winner: Add green outline to more consistent system (default: True)
        height: Plot height in pixels (default: 400)

    Returns:
        plotly.graph_objects.Figure or None
    """

    # Extract full records including CFR data
    cols_needed = [year_col, week_col, 'TotalCases', 'Deaths', 'CFR']

    llm_timeline = batch_df[
        (batch_df['Country'] == country) &
        (batch_df['Event'] == event)
    ][cols_needed].copy()
    llm_timeline = llm_timeline.drop_duplicates().sort_values([year_col, week_col])

    baseline_timeline = ruleb_df[
        (ruleb_df['Country'] == country) &
        (ruleb_df['Event'] == event)
    ][cols_needed].copy()
    baseline_timeline = baseline_timeline.drop_duplicates().sort_values([year_col, week_col])

    if len(llm_timeline) == 0 and len(baseline_timeline) == 0:
        return None

    # Calculate CFR consistency
    if highlight_cfr_winner:
        llm_timeline = check_cfr_consistency(llm_timeline, year_col, week_col)
        baseline_timeline = check_cfr_consistency(baseline_timeline, year_col, week_col)

    # Create time labels (YYYY-WNN format)
    llm_timeline['TimeLabel'] = (
        llm_timeline[year_col].astype(str) + '-W' +
        llm_timeline[week_col].astype(str).str.zfill(2)
    )
    baseline_timeline['TimeLabel'] = (
        baseline_timeline[year_col].astype(str) + '-W' +
        baseline_timeline[week_col].astype(str).str.zfill(2)
    )

    # Merge to find discrepancies (where values differ)
    merged = llm_timeline.merge(
        baseline_timeline,
        on=['TimeLabel', year_col, week_col],
        how='outer',
        suffixes=('_llm', '_baseline')
    )
    merged['has_discrepancy'] = (
        merged['TotalCases_llm'].fillna(-1) != merged['TotalCases_baseline'].fillna(-1)
    )

    # Determine which system has better CFR consistency at each point
    if highlight_cfr_winner:
        merged['llm_better_cfr'] = (
            merged['cfr_error_llm'].fillna(999) < merged['cfr_error_baseline'].fillna(999)
        )
        merged['baseline_better_cfr'] = (
            merged['cfr_error_baseline'].fillna(999) < merged['cfr_error_llm'].fillna(999)
        )

    # Create figure
    fig = go.Figure()

    if len(llm_timeline) > 0:
        # Determine marker color (lime fill where LLM has better CFR + discrepancy)
        if highlight_cfr_winner:
            marker_colors = []
            marker_sizes = []
            marker_line_colors = []
            marker_line_widths = []
            for _, row in llm_timeline.iterrows():
                time_label = row['TimeLabel']
                disc_info = merged[merged['TimeLabel'] == time_label].iloc[0] if len(merged[merged['TimeLabel'] == time_label]) > 0 else None

                if disc_info is not None and disc_info['has_discrepancy'] and disc_info['llm_better_cfr']:
                    marker_colors.append('lime')  # Bright lime for winner
                    marker_sizes.append(16)  # Bigger
                    marker_line_colors.append('darkgreen')
                    marker_line_widths.append(3)
                else:
                    marker_colors.append('blue')
                    marker_sizes.append(10)  # Smaller
                    marker_line_colors.append('darkblue')
                    marker_line_widths.append(1)

            marker_dict = dict(
                size=marker_sizes,  # Variable size
                symbol='circle',
                color=marker_colors,
                line=dict(color=marker_line_colors, width=marker_line_widths)
            )
        else:
            marker_dict = dict(size=8, symbol='circle', color='blue')

        fig.add_trace(go.Scatter(
            name='LLM',
            x=llm_timeline['TimeLabel'],
            y=llm_timeline['TotalCases'],
            mode='lines+markers',
            marker=marker_dict,
            line=dict(color='blue', width=2),
            hovertemplate='%{y:,.0f} cases<extra></extra>'
        ))

    if len(baseline_timeline) > 0:
        # Determine marker color (lime fill where Baseline has better CFR + discrepancy)
        if highlight_cfr_winner:
            marker_colors = []
            marker_sizes = []
            marker_line_colors = []
            marker_line_widths = []
            for _, row in baseline_timeline.iterrows():
                time_label = row['TimeLabel']
                disc_info = merged[merged['TimeLabel'] == time_label].iloc[0] if len(merged[merged['TimeLabel'] == time_label]) > 0 else None

                if disc_info is not None and disc_info['has_discrepancy'] and disc_info['baseline_better_cfr']:
                    marker_colors.append('lime')  # Bright lime for winner
                    marker_sizes.append(16)  # Bigger
                    marker_line_colors.append('darkgreen')
                    marker_line_widths.append(3)
                else:
                    marker_colors.append('orange')
                    marker_sizes.append(10)  # Smaller
                    marker_line_colors.append('darkorange')
                    marker_line_widths.append(1)

            marker_dict = dict(
                size=marker_sizes,  # Variable size
                symbol='square',
                color=marker_colors,
                line=dict(color=marker_line_colors, width=marker_line_widths)
            )
        else:
            marker_dict = dict(size=8, symbol='square', color='orange')

        fig.add_trace(go.Scatter(
            name='Baseline',
            x=baseline_timeline['TimeLabel'],
            y=baseline_timeline['TotalCases'],
            mode='lines+markers',
            marker=marker_dict,
            line=dict(color='orange', width=2, dash='dash'),
            hovertemplate='%{y:,.0f} cases<extra></extra>'
        ))

    title_suffix = ' (lime markers = better CFR consistency)' if highlight_cfr_winner else ''
    fig.update_layout(
        title=f'{country} - {event}{title_suffix}',
        xaxis_title='Time (Year-Week)',
        yaxis_title='Total Cases',
        hovermode='x unified',
        height=height,
        showlegend=True,
        xaxis=dict(tickangle=45)
    )

    return fig


def create_individual_timeline_plots(disc_cat, batch_df, ruleb_df,
                                      parameter='TotalCases', n_top=None,
                                      week_col='WeekNumber', year_col='Year',
                                      highlight_cfr_winner=True, height=400):
    """
    Create individual timeline plots (NOT subplots) for discrepancies.

    Better for notebook rendering than subplots.

    Args:
        disc_cat: Categorized discrepancies dataframe
        batch_df: LLM batch extraction dataframe
        ruleb_df: Rule-based baseline dataframe
        parameter: Which parameter to visualize (default: 'TotalCases')
        n_top: Number of top discrepancies to show (default: None = all)
        week_col: Name of week column (default: 'WeekNumber')
        year_col: Name of year column (default: 'Year')
        highlight_cfr_winner: Add lime highlighting (default: True)
        height: Height per plot (default: 400)

    Returns:
        List of plotly.graph_objects.Figure objects
    """

    # Filter for specific parameter discrepancies
    param_disc = disc_cat[disc_cat['Parameter'] == parameter].copy()

    if len(param_disc) == 0:
        print(f"No {parameter} discrepancies found!")
        return []

    # Get unique country-event pairs
    country_event_pairs = param_disc[['Country', 'Event']].drop_duplicates()

    # Calculate max discrepancy for each pair
    disc_summary = []
    for _, row in country_event_pairs.iterrows():
        country = row['Country']
        event = row['Event']
        pair_disc = param_disc[
            (param_disc['Country'] == country) &
            (param_disc['Event'] == event)
        ]

        if len(pair_disc) > 0:
            # Get max absolute difference
            max_disc = pair_disc.apply(
                lambda x: abs(
                    float(str(x['LLM']).replace(',', '') or 0) -
                    float(str(x['Baseline']).replace(',', '') or 0)
                ),
                axis=1
            ).max()

            disc_summary.append({
                'country': country,
                'event': event,
                'max_discrepancy': max_disc
            })

    # Sort by discrepancy severity
    disc_summary_df = pd.DataFrame(disc_summary).sort_values(
        'max_discrepancy', ascending=False
    )

    # Take top N or all if n_top is None
    if n_top is not None:
        selected_pairs = disc_summary_df.head(n_top)
    else:
        selected_pairs = disc_summary_df

    # Create individual plots
    figures = []
    for idx, (_, row) in enumerate(selected_pairs.iterrows(), 1):
        country = row['country']
        event = row['event']

        print(f"Creating plot {idx}/{len(selected_pairs)}: {country} - {event}")

        fig = create_timeline_plot(
            batch_df, ruleb_df, country, event,
            week_col, year_col,
            highlight_cfr_winner=highlight_cfr_winner,
            height=height
        )

        if fig:
            figures.append(fig)

    return figures


def create_top_discrepancy_plots(disc_cat, batch_df, ruleb_df,
                                  parameter='TotalCases', n_top=10,
                                  week_col='WeekNumber', year_col='Year',
                                  highlight_cfr_winner=True):
    """
    Create a combined subplot visualization of top N discrepancies.

    Green outlines highlight which system has better CFR consistency at each point.

    Args:
        disc_cat: Categorized discrepancies dataframe
        batch_df: LLM batch extraction dataframe
        ruleb_df: Rule-based baseline dataframe
        parameter: Which parameter to visualize (default: 'TotalCases')
        n_top: Number of top discrepancies to show (default: 10)
        week_col: Name of week column (default: 'WeekNumber')
        year_col: Name of year column (default: 'Year')
        highlight_cfr_winner: Add green outline to more consistent system (default: True)

    Returns:
        plotly.graph_objects.Figure
    """

    # Filter for specific parameter discrepancies
    param_disc = disc_cat[disc_cat['Parameter'] == parameter].copy()

    if len(param_disc) == 0:
        print(f"No {parameter} discrepancies found!")
        return None

    # Get unique country-event pairs
    country_event_pairs = param_disc[['Country', 'Event']].drop_duplicates()

    # Calculate max discrepancy for each pair
    disc_summary = []
    for _, row in country_event_pairs.iterrows():
        country = row['Country']
        event = row['Event']
        pair_disc = param_disc[
            (param_disc['Country'] == country) &
            (param_disc['Event'] == event)
        ]

        if len(pair_disc) > 0:
            # Get max absolute difference
            max_disc = pair_disc.apply(
                lambda x: abs(
                    float(str(x['LLM']).replace(',', '') or 0) -
                    float(str(x['Baseline']).replace(',', '') or 0)
                ),
                axis=1
            ).max()

            disc_summary.append({
                'country': country,
                'event': event,
                'max_discrepancy': max_disc
            })

    # Get top N
    disc_summary_df = pd.DataFrame(disc_summary).sort_values(
        'max_discrepancy', ascending=False
    )
    top_n = disc_summary_df.head(n_top)

    # Create subplots (2 columns)
    n_cols = 2
    n_rows = (len(top_n) + n_cols - 1) // n_cols

    subplot_titles = [
        f"{row['country']} - {row['event']}"
        for _, row in top_n.iterrows()
    ]

    fig_combined = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,  # More space between rows
        horizontal_spacing=0.15  # More space between columns
    )

    # Add traces for each top discrepancy
    for idx, (_, row) in enumerate(top_n.iterrows()):
        country = row['country']
        event = row['event']
        row_num = (idx // n_cols) + 1
        col_num = (idx % n_cols) + 1

        # Create individual plot with CFR highlighting
        individual_fig = create_timeline_plot(
            batch_df, ruleb_df, country, event, week_col, year_col,
            highlight_cfr_winner=highlight_cfr_winner
        )

        if individual_fig:
            for trace in individual_fig.data:
                # Explicitly preserve all trace properties including per-point marker properties
                # Convert tuples to lists for plotly compatibility
                marker_size = list(trace.marker.size) if isinstance(trace.marker.size, (list, tuple)) else trace.marker.size
                marker_color = list(trace.marker.color) if isinstance(trace.marker.color, (list, tuple)) else trace.marker.color
                marker_line_color = list(trace.marker.line.color) if trace.marker.line and isinstance(trace.marker.line.color, (list, tuple)) else (trace.marker.line.color if trace.marker.line else None)
                marker_line_width = list(trace.marker.line.width) if trace.marker.line and isinstance(trace.marker.line.width, (list, tuple)) else (trace.marker.line.width if trace.marker.line else None)

                trace_copy = go.Scatter(
                    x=trace.x,
                    y=trace.y,
                    mode=trace.mode,
                    name=trace.name,
                    marker=dict(
                        size=marker_size,
                        symbol=trace.marker.symbol,
                        color=marker_color,
                        line=dict(
                            color=marker_line_color,
                            width=marker_line_width
                        ) if marker_line_color is not None else None
                    ),
                    line=dict(
                        color=trace.line.color,
                        width=trace.line.width,
                        dash=trace.line.dash if hasattr(trace.line, 'dash') else None
                    ) if trace.line else None,
                    hovertemplate=trace.hovertemplate,
                    showlegend=(idx == 0)  # Only show legend once
                )
                fig_combined.add_trace(trace_copy, row=row_num, col=col_num)

    # Update layout
    title_suffix = " (lime markers = better CFR consistency)" if highlight_cfr_winner else ""
    fig_combined.update_layout(
        title_text=f"Top {len(top_n)} {parameter} Discrepancies: LLM vs Baseline{title_suffix}",
        height=500 * n_rows,  # Taller plots
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        font=dict(size=10)  # Smaller font for better fit
    )

    # Update axes with better formatting
    for i in range(1, len(top_n) + 1):
        row_num = (i-1)//n_cols+1
        col_num = (i-1)%n_cols+1

        fig_combined.update_yaxes(
            title_text="Cases",
            row=row_num,
            col=col_num
        )

        fig_combined.update_xaxes(
            tickangle=45,  # Angle x-axis labels
            row=row_num,
            col=col_num
        )

    return fig_combined
