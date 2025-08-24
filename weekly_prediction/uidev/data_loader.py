import os
import pandas as pd
import streamlit as st


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _read_csv_if_exists(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_ensemble_weekly() -> pd.DataFrame:
    csv_path = os.path.join(BASE_DIR, "data", "ensemble", "ensemble_weekly_output.csv")
    df = _read_csv_if_exists(csv_path)
    # Normalize common column names
    rename_map = {
        "xgb_pred": "xgb_predicted_return_pct",
        "lstm_pred": "lstm_predicted_return_pct",
        "confidence_score_xgb": "xgb_confidence_score",
        "confidence_score_lstm": "lstm_confidence_score",
    }
    df = df.rename(columns=rename_map)
    return df
@st.cache_data(show_spinner=False)
def load_ui_unified() -> pd.DataFrame:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    csv_path = os.path.join(base, "data", "features", "ui_unified_features.csv")
    df = _read_csv_if_exists(csv_path)
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype(str).str.upper()
    return df



@st.cache_data(show_spinner=False)
def load_xgboost_weekly() -> pd.DataFrame:
    csv_path = os.path.join(BASE_DIR, "data", "xgboost", "xgboost_weekly_output.csv")
    df = _read_csv_if_exists(csv_path)
    return df


@st.cache_data(show_spinner=False)
def load_lstm_weekly() -> pd.DataFrame:
    csv_path = os.path.join(BASE_DIR, "data", "lstm", "lstm_weekly_output.csv")
    df = _read_csv_if_exists(csv_path)
    # Normalize column names to align with ensemble/xgb pages where helpful
    rename_map = {
        "confidence_score": "lstm_confidence_score",
        "confidence_score_adj": "lstm_confidence_score_adj",
        "lstm_pred_adj": "lstm_predicted_return_pct_adj",
    }
    df = df.rename(columns=rename_map)
    return df


def _numeric_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def default_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in [
        "ensemble_score",
        "xgb_score_adj",
        "xgb_score",
        "xgb_predicted_return_pct",
        "lstm_predicted_return_pct",
    ]:
        if col in df.columns:
            return df.sort_values(col, ascending=False)
    # Fallback: sort by first numeric column desc
    nums = _numeric_columns(df)
    return df.sort_values(nums[0], ascending=False) if nums else df


