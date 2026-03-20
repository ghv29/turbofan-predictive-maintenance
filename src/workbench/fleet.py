from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .loader import load_combined_tableau, load_engineered_dataset
from .predictor import predict_rul_for_rows, shap_for_single_row
from .utils import HealthStatus, degradation_slope, rul_to_health_status


@dataclass(frozen=True)
class FleetSummary:
    total_engines: int
    critical_count: int
    warning_count: int
    healthy_count: int


@st.cache_data(show_spinner=False)
def compute_fleet_latest_snapshot() -> Tuple[pd.DataFrame, FleetSummary]:
    """
    Compute a single latest-cycle snapshot row per engine across all datasets,
    with a model-based RUL prediction and health status.

    Returns
    -------
    latest_df : DataFrame
        Columns: engine_id, dataset, cycle, RUL_true, predicted_RUL, health_status
    summary : FleetSummary
        Aggregate counts by health band.
    """
    df = load_combined_tableau()

    # Latest cycle per engine_id + dataset
    idx = df.groupby(["dataset", "engine_id"])["cycle"].idxmax()
    latest = df.loc[idx, ["dataset", "engine_id", "cycle", "RUL"]].copy()
    latest = latest.rename(columns={"RUL": "RUL_true"})

    # Join with engineered/scaled features where available
    frames: List[pd.DataFrame] = []
    for dataset, group in latest.groupby("dataset"):
        engineered = load_engineered_dataset(dataset)
        if engineered is None:
            # Fall back to using true RUL and inferred health status only.
            g = group.copy()
            g["predicted_RUL"] = g["RUL_true"]
            g["health_status"] = [
                rul_to_health_status(r) for r in g["predicted_RUL"].to_numpy()
            ]
            frames.append(g)
            continue

        merged = pd.merge(
            group,
            engineered,
            on=["engine_id", "cycle"],
            how="left",
            suffixes=("", "_feat"),
        )
        feature_cols = [c for c in engineered.columns if c not in {"engine_id", "cycle", "RUL"}]
        X = merged[feature_cols]
        preds = predict_rul_for_rows(X)
        merged["predicted_RUL"] = preds
        merged["health_status"] = [
            rul_to_health_status(v) for v in preds
        ]
        frames.append(merged[["dataset", "engine_id", "cycle", "RUL_true", "predicted_RUL", "health_status"]])

    latest_all = pd.concat(frames, ignore_index=True)

    critical = (latest_all["health_status"] == "CRITICAL").sum()
    warning = (latest_all["health_status"] == "WARNING").sum()
    healthy = (latest_all["health_status"] == "HEALTHY").sum()

    summary = FleetSummary(
        total_engines=len(latest_all),
        critical_count=int(critical),
        warning_count=int(warning),
        healthy_count=int(healthy),
    )
    return latest_all, summary


@st.cache_data(show_spinner=False)
def dataset_level_stats(latest_snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average predicted RUL per dataset for the bar chart.
    """
    return (
        latest_snapshot.groupby("dataset")["predicted_RUL"]
        .mean()
        .reset_index(name="avg_predicted_RUL")
    )


@st.cache_data(show_spinner=False)
def rul_distribution(latest_snapshot: pd.DataFrame) -> np.ndarray:
    """
    Convenience helper returning the array of predicted RULs for histograms.
    """
    return latest_snapshot["predicted_RUL"].to_numpy()


@st.cache_data(show_spinner=False)
def top_critical_engines(
    latest_snapshot: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    """
    Return the top N most critical engines (lowest RUL) with basic fields.
    """
    df = latest_snapshot.copy()
    df = df.sort_values("predicted_RUL").head(top_n)
    return df.reset_index(drop=True)

