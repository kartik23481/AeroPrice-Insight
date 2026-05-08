# docker_backend/inference/preprocess.py

import pandas as pd

def preprocess_input(user_input: dict) -> pd.DataFrame:
    """
    Converting API input into model-ready DataFrame.
    """

    df = pd.DataFrame([user_input]).copy()

    # normalising string columns
    for col in ["airline", "source", "destination", "additional_info"]:
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()

    if "date" not in df.columns:
        raise ValueError("Missing required date field from API input")

    date = pd.to_datetime(df["date"])

    df["dtoj_day"] = date.dt.day
    df["dtoj_month"] = date.dt.month
    df["dtoj_year"] = date.dt.year
    df["is_weekend"] = (date.dt.weekday >= 5).astype(int)

    # removing date
    df = df.drop(columns=["date"], errors="ignore")

    df = df.drop(columns=["dep_time_min"], errors="ignore")

    return df
