
# # docker_backend/inference/preprocess.py

# import pandas as pd

# def preprocess_input(user_input: dict) -> pd.DataFrame:
#     """
#     Convert API input into model-ready DataFrame
#     Must exactly match training-time schema
#     """

#     df = pd.DataFrame([user_input])

#     # ---- text normalization (same as training) ----
#     for col in ["airline", "source", "destination", "additional_info"]:
#         df[col] = df[col].str.lower().str.strip()

#     # ---- date handling ----
#     df["date"] = pd.to_datetime(df["date"])

#     df["dtoj_day"] = df["date"].dt.day
#     df["dtoj_month"] = df["date"].dt.month
#     df["dtoj_year"] = df["date"].dt.year
#     df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(int)

#     return df



# docker_backend/inference/preprocess.py

import pandas as pd

def preprocess_input(user_input: dict) -> pd.DataFrame:
    """
    Convert API input into model-ready DataFrame.
    Must exactly match training-time preprocess.
    """

    df = pd.DataFrame([user_input]).copy()

    # normalize string columns
    for col in ["airline", "source", "destination", "additional_info"]:
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()

    # ==============================================
    # HANDLE DATE (raw → engineered)
    # same as training
    # ==============================================
    if "date" not in df.columns:
        raise ValueError("Missing required date field from API input")

    date = pd.to_datetime(df["date"])

    df["dtoj_day"] = date.dt.day
    df["dtoj_month"] = date.dt.month
    df["dtoj_year"] = date.dt.year
    df["is_weekend"] = (date.dt.weekday >= 5).astype(int)

    # remove raw date
    df = df.drop(columns=["date"], errors="ignore")

    # safe drop
    df = df.drop(columns=["dep_time_min"], errors="ignore")

    return df
