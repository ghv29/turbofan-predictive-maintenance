from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .utils import health_color


def fleet_health_grid(latest_snapshot: pd.DataFrame) -> go.Figure:
    """
    Create a scatter-based grid: one point per engine, colored by health.

    Layout engines along the x-axis in engine index order and datasets on y.
    Hover shows engine id, dataset, and RUL.
    """
    df = latest_snapshot.copy()
    df = df.sort_values(["dataset", "engine_id"])
    df["engine_index"] = range(1, len(df) + 1)

    fig = px.scatter(
        df,
        x="engine_index",
        y="dataset",
        color="health_status",
        color_discrete_map={
            "CRITICAL": health_color("CRITICAL"),
            "WARNING": health_color("WARNING"),
            "HEALTHY": health_color("HEALTHY"),
        },
        hover_data={
            "engine_id": True,
            "dataset": True,
            "predicted_RUL": ":.1f",
            "engine_index": False,
        },
    )
    fig.update_traces(marker=dict(size=14, symbol="square"))
    fig.update_layout(
        xaxis_title="Engine index (sorted by dataset and ID)",
        yaxis_title="Dataset",
        legend_title="Health status",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def dataset_comparison_bar(stats: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart of average predicted RUL per dataset.
    """
    fig = px.bar(
        stats,
        x="dataset",
        y="avg_predicted_RUL",
        text="avg_predicted_RUL",
        color_discrete_sequence=[health_color("HEALTHY")],
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        yaxis_title="Average predicted RUL (cycles)",
        xaxis_title="Dataset",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def rul_distribution_hist(ruls: np.ndarray) -> go.Figure:
    """
    Histogram of predicted RUL across the fleet with threshold markers.
    """
    fig = go.Figure()
    fig.add_histogram(
        x=ruls,
        nbinsx=40,
        marker_color=health_color("HEALTHY"),
        opacity=0.7,
    )

    for x, label, color in [
        (30, "Critical threshold (30)", health_color("CRITICAL")),
        (70, "Warning threshold (70)", health_color("WARNING")),
    ]:
        fig.add_vline(
            x=x,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top right",
        )

    fig.update_layout(
        xaxis_title="Predicted RUL (cycles)",
        yaxis_title="Count of engines",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def degradation_timeline(
    df_top_features: pd.DataFrame,
    rul_trajectory: pd.DataFrame,
    current_cycle: int,
) -> go.Figure:
    """
    Plot top 3 SHAP features (already extracted over time) and predicted RUL.
    """
    fig = go.Figure()

    # Left axis: sensor trends
    for col in df_top_features.columns:
        if col == "cycle":
            continue
        fig.add_trace(
            go.Scatter(
                x=df_top_features["cycle"],
                y=df_top_features[col],
                mode="lines",
                name=col,
            )
        )

    # Right axis: predicted RUL
    fig.add_trace(
        go.Scatter(
            x=rul_trajectory["cycle"],
            y=rul_trajectory["predicted_RUL"],
            mode="lines+markers",
            name="Predicted RUL",
            yaxis="y2",
            line=dict(color=health_color("HEALTHY"), dash="dash"),
        )
    )

    # Current cycle marker
    fig.add_vline(
        x=current_cycle,
        line_dash="dash",
        line_color="#AAAAAA",
        annotation_text="Current cycle",
        annotation_position="top right",
    )

    # Shade future region
    max_cycle = max(df_top_features["cycle"].max(), rul_trajectory["cycle"].max())
    fig.add_vrect(
        x0=current_cycle,
        x1=max_cycle,
        line_width=0,
        fillcolor="#2c3e50",
        opacity=0.15,
    )

    fig.update_layout(
        xaxis_title="Cycle",
        yaxis_title="Sensor value (normalized)",
        yaxis2=dict(
            title="Predicted RUL (cycles)",
            overlaying="y",
            side="right",
        ),
        legend_title="Signals",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def shap_top_features_bar(shap_values: Dict[str, float], top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart of top-N absolute SHAP values.
    """
    items = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    features = [name for name, _ in items]
    values = [val for _, val in items]

    colors = [health_color("HEALTHY") if v >= 0 else health_color("CRITICAL") for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        xaxis_title="SHAP value (impact on RUL prediction)",
        yaxis_title="Feature",
        margin=dict(l=80, r=20, t=40, b=40),
    )
    return fig

