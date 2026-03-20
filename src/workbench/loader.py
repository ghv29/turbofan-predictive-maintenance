from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
import streamlit as st

from .utils import HealthStatus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


@st.cache_resource(show_spinner=False)
def load_model_and_features() -> Tuple[object, Tuple[str, ...]]:
    """
    Load the trained RandomForest model and its feature column order.

    Model path is fixed by project spec. Feature column order is stored
    in `feature_cols.pkl`. If either file is missing, this function
    raises a clear RuntimeError so the UI can display instructions.
    """
    model_path = MODELS_DIR / "best_random_forest.pkl"
    feature_cols_path = MODELS_DIR / "feature_cols.pkl"

    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found at {model_path}. "
            "Please ensure `best_random_forest.pkl` is saved under `outputs/models/`."
        )
    if not feature_cols_path.exists():
        raise RuntimeError(
            f"Feature column file not found at {feature_cols_path}. "
            "Expected a joblib/PKL containing the 44 feature names."
        )

    model = joblib.load(model_path)
    feature_cols = tuple(joblib.load(feature_cols_path))
    return model, feature_cols


@st.cache_data(show_spinner=False)
def load_combined_tableau() -> pd.DataFrame:
    """
    Load the combined `tableau_all_datasets.csv` backbone.

    This file contains raw sensor readings, engine_id, cycle, dataset,
    RUL, and health_status for all four FD00x datasets.
    """
    path = DATA_PROCESSED_DIR / "tableau_all_datasets.csv"
    if not path.exists():
        raise RuntimeError(
            f"Combined dataset file not found at {path}. "
            "Please export `tableau_all_datasets.csv` from the notebooks."
        )

    df = pd.read_csv(path)
    # Ensure expected identifier columns exist
    required_cols = {"engine_id", "cycle", "dataset", "RUL"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"`tableau_all_datasets.csv` missing columns: {missing}")
    return df


ENGINEERED_FILENAMES: Dict[str, str] = {
    "FD001": "train_FD001_processed.csv",
    "FD002": "train_FD002_processed.csv",
    "FD003": "train_FD003_processed.csv",
    "FD004": "train_FD004_processed.csv",
}


@st.cache_data(show_spinner=False)
def load_engineered_dataset(dataset: str) -> Optional[pd.DataFrame]:
    """
    Load the engineered + scaled features for a given FD00x dataset.

    Returns None if the file is missing so the caller can degrade gracefully.
    """
    filename = ENGINEERED_FILENAMES.get(dataset)
    if not filename:
        return None

    path = DATA_PROCESSED_DIR / filename
    if not path.exists():
        return None

    df = pd.read_csv(path)
    # Expect at minimum the identifiers + RUL to be present.
    required = {"engine_id", "cycle", "RUL"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{filename} missing required columns: {missing}")
    return df


@lru_cache(maxsize=1)
def available_datasets() -> Tuple[str, ...]:
    """
    Return the FD00x dataset labels that have engineered feature files present.
    """
    present = []
    for ds, fname in ENGINEERED_FILENAMES.items():
        if (DATA_PROCESSED_DIR / fname).exists():
            present.append(ds)
    return tuple(sorted(present))

