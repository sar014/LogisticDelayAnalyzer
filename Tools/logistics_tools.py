# logistics_tools.py
from crewai.tools import tool

# This assumes df is provided from outside; you'll wire this later
df = None

def set_dataframe(global_df):
    global df
    df = global_df

@tool("delay_column_stats")
def delay_column_stats(column_name: str) -> str:
    """
    Return basic stats for a given delay-related column in the logistics dataframe.
    """
    if df is None:
        return "Dataframe not initialized in tool."

    if column_name not in df.columns:
        return f"Column '{column_name}' not found."

    series = df[column_name].dropna()
    if series.empty:
        return f"Column '{column_name}' has no data."

  
    return (
        f"Stats for {column_name}: "
        f"count={len(series)}, "
        f"unique={series.nunique()}, "
        f"sample_values={series.value_counts().head(5).to_dict()}"
    )
# logistics_tools.py
import pandas as pd
from crewai.tools import tool

_df: pd.DataFrame | None = None

def set_dataframe(df: pd.DataFrame):
    global _df
    _df = df

def _require_df() -> pd.DataFrame:
    if _df is None:
        raise ValueError("DataFrame not set. Call set_dataframe(df) first.")
    return _df

@tool
def get_dataset_profile() -> dict:
    """
    Return basic profile of the logistics dataset.
    """
    df = _require_df()
    return {
        "total_rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records"),
    }

@tool
def find_delay_columns() -> dict:
    """
    Heuristically guess delay indicator columns.
    """
    df = _require_df()
    delay_like = []
    for col in df.columns:
        name = col.lower()
        if any(k in name for k in ["delay", "lateness", "late", "sla_violation"]):
            delay_like.append(col)
    return {"delay_candidates": delay_like}

@tool
def compute_delay_stats(delay_col: str) -> dict:
    """
    Compute delay stats for the given delay indicator column.
    """
    df = _require_df()
    if delay_col not in df.columns:
        return {"error": f"Column {delay_col} not in dataframe"}
    series = df[delay_col]
    value_counts = series.value_counts(dropna=False).to_dict()
    total = int(len(series))
    delayed_values = {k: v for k, v in value_counts.items()
                      if str(k).lower() in ["yes", "y", "true", "1", "delayed"]}
    delayed_count = sum(delayed_values.values()) if delayed_values else 0
    delayed_pct = delayed_count / total if total else 0.0
    return {
        "delay_column": delay_col,
        "value_counts": {str(k): int(v) for k, v in value_counts.items()},
        "delayed_count": int(delayed_count),
        "delayed_pct": float(delayed_pct),
        "total": total,
    }

@tool
def list_delay_factors(factor_cols: list[str]) -> dict:
    """
    For explicit factor columns (e.g. Delay_Reason),
    return top categories and counts.
    """
    df = _require_df()
    out = {}
    for col in factor_cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False).head(10).to_dict()
        out[col] = {str(k): int(v) for k, v in vc.items()}
    return {"factor_stats": out}
