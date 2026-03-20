from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .loader import load_model_and_features, load_engineered_dataset
from .utils import HealthStatus, rul_to_health_status


def predict_rul_for_rows(df_features: pd.DataFrame) -> np.ndarray:
    """
    Predict RUL for a batch of feature rows using the trained model.

    `df_features` must already be in the correct feature order and
    scaled, matching `feature_cols.pkl`.
    """
    model, feature_cols = load_model_and_features()
    # Keep feature names/order by passing a DataFrame to sklearn.
    # This avoids the scikit-learn warning about "valid feature names".
    X = df_features.loc[:, list(feature_cols)]
    preds = model.predict(X)
    # Enforce RUL cap at 125 cycles as per project spec.
    preds = np.clip(preds, 0, 125)
    return preds


def predict_rul_with_status(df_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return predicted RUL and corresponding health status codes.
    """
    rul = predict_rul_for_rows(df_features)
    status = np.array([rul_to_health_status(v) for v in rul])
    return rul, status


def shap_for_single_row(dataset: str, engine_id: int, cycle: int) -> Dict[str, float]:
    """
    Compute SHAP values for a single engine at a specific cycle.

    Returns a mapping {feature_name: shap_value} for the model's
    current prediction on that row.
    """
    engineered = load_engineered_dataset(dataset)
    if engineered is None:
        raise RuntimeError(f"No engineered features available for dataset {dataset}.")

    model, feature_cols = load_model_and_features()

    row = engineered[
        (engineered["engine_id"] == engine_id) & (engineered["cycle"] == cycle)
    ]
    if row.empty:
        raise RuntimeError(
            f"No engineered row found for engine {engine_id}, cycle {cycle} in {dataset}."
        )

    X_row = row.loc[:, list(feature_cols)]

    try:
        import shap  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is not installed in this environment. "
            "Install it to enable explanation charts."
        ) from exc

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)[0]  # random forest: array of shape (n_features,)

    return {feat: float(val) for feat, val in zip(feature_cols, shap_values)}


def rul_trajectory_for_engine(dataset: str, engine_id: int) -> pd.DataFrame:
    """
    For a given engine, compute predicted RUL at each cycle using
    the engineered feature rows.

    Returns a DataFrame with columns: [cycle, predicted_RUL, health_status].
    """
    engineered = load_engineered_dataset(dataset)
    if engineered is None:
        raise RuntimeError(f"No engineered features available for dataset {dataset}.")

    subset = engineered[engineered["engine_id"] == engine_id].copy()
    if subset.empty:
        raise RuntimeError(f"Engine {engine_id} not found in engineered data for {dataset}.")

    # Sort by cycle to ensure consistent trajectory
    subset = subset.sort_values("cycle")

    model, feature_cols = load_model_and_features()
    X = subset.loc[:, list(feature_cols)]
    # Passing DataFrame preserves feature names and avoids warnings.
    preds = model.predict(X)
    preds = np.clip(preds, 0, 125)

    health = [rul_to_health_status(v) for v in preds]

    out = pd.DataFrame(
        {
            "cycle": subset["cycle"].to_numpy(),
            "predicted_RUL": preds,
            "health_status": health,
        }
    )
    return out

